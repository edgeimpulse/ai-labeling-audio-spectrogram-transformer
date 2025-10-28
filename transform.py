import os
import sys
import requests
import time
import argparse
import json
import math
import base64
# Import the API objects we plan to use
from edgeimpulse_api import ApiClient, Configuration, ProjectsApi, RawDataApi
from edgeimpulse_api.models.set_sample_structured_labels_request import (
    SetSampleStructuredLabelsRequest,
)
from edgeimpulse_api.models.edit_sample_label_request import (
    EditSampleLabelRequest,
)
# For splitting audio
from pydub import AudioSegment

# Set and retrieve env. variables
if not os.getenv("EI_PROJECT_API_KEY"):
    print("Missing EI_PROJECT_API_KEY")
    sys.exit(1)
if not os.getenv("BEAM_ENDPOINT"):
    print("Missing BEAM_ENDPOINT")
    sys.exit(1)
if not os.getenv("BEAM_ACCESS_KEY"):
    print("Missing BEAM_ACCESS_KEY")
    sys.exit(1)

EI_PROJECT_API_KEY = os.environ.get("EI_PROJECT_API_KEY")
EI_API_ENDPOINT = os.environ.get("EI_API_ENDPOINT", "https://studio.edgeimpulse.com/v1")
BEAM_ENDPOINT = os.environ.get("BEAM_ENDPOINT")
BEAM_ACCESS_KEY = os.environ.get("BEAM_ACCESS_KEY")
OUTPUT_PATH = "./out"

# Argument parser for command line arguments
parser = argparse.ArgumentParser(
    description="Use Beam.cloud to classify sound samples in your dataset"
)
parser.add_argument(
    "--audioset-labels",
    type=str,
    required=True,
    help='Comma-separated list of labels from "AudioSet" that will be used to label the sample.',
)
parser.add_argument(
    "--win-size-ms",
    type=int,
    required=True,
    help="Size of the window for each classification",
)
parser.add_argument(
    "--win-stride-ms",
    type=int,
    required=True,
    help="Stride of the window for each classification",
)
parser.add_argument("--other-label", type=str, default='other',
    help='Other label')
parser.add_argument("--min-confidence", type=float, default=0.2,
    help='Classifications below the threshold are discarded')
parser.add_argument("--data-ids-file", type=str, required=True,
    help='File with IDs (as JSON)')
parser.add_argument("--propose-actions", type=int, required=False,
    help='If this flag is passed in, only propose suggested actions')
args, unknown = parser.parse_known_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, 'labels.txt'), 'r') as f:
    valid_labels = [x.strip().lower() for x in f.read().split("\n")]

audioset_labels = args.audioset_labels.replace('\\n', '\n')
audioset_labels_list = {}
for audioset_label in [x.strip().lower() for x in audioset_labels.strip().split("\n")]:
    if audioset_label in valid_labels:
        audioset_labels_list[audioset_label] = audioset_label
        continue
    if '(' in audioset_label and ')' in audioset_label:
        remapped_label = audioset_label[audioset_label.rindex('(') + 1:audioset_label.rindex(')')]
        filtered_label = audioset_label[0:audioset_label.rindex('(')].strip()
        if filtered_label in valid_labels:
            audioset_labels_list[filtered_label] = remapped_label
            continue
    print('Valid labels: ' + ', '.join(valid_labels))
    print('')
    print('Invalid label: "' + audioset_label + '" (see above for list of valid labels)')
    exit(1)

audioset_labels_list_str = ', '.join([(k if audioset_labels_list[k] == k else k + ' (' + audioset_labels_list[k] + ')') for k in audioset_labels_list.keys()])
win_size_ms = args.win_size_ms
win_stride_ms = args.win_stride_ms

if args.data_ids_file:
    with open(args.data_ids_file, 'r') as f:
        data_ids = json.load(f)
other_label = args.other_label
min_confidence = args.min_confidence

if win_stride_ms > win_size_ms:
    print('ERR: Window size needs to be the same size or bigger than window stride')
    exit(1)

print('Labeling data using Audio Spectrogram Transformers')
print('')
print('Detecting audio:')
print('    Audioset labels:', audioset_labels_list_str)
print('    Other label:', other_label)
print('    Min. confidence:', min_confidence)
print(f"    Window size: {win_size_ms}ms.")
print(f"    Window stride: {win_stride_ms}ms.")
if len(data_ids) < 6:
    print('    IDs:', ', '.join([str(x) for x in data_ids]))
else:
    print('    IDs:', ', '.join([str(x) for x in data_ids[0:5]]), 'and ' + str(len(data_ids) - 5) + ' others')
print('')

# NEW: Function to call Beam.cloud endpoint
def classify_audio_sample(filename: str, beam_endpoint: str, beam_access_key: str, labels: list[str]):
    with open(filename, "rb") as f:
        base64_audio = base64.b64encode(f.read()).decode("utf-8")
    body = json.dumps({
        "base64_audio": base64_audio,
        "labels": list(audioset_labels_list.keys()),
    })
    headers = {
        "Authorization": f"Bearer {beam_access_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(beam_endpoint, headers=headers, data=body)
    if response.status_code != 200:
        raise Exception(f"Failed to classify audio: {response.text}")
    return response.json()["predictions"]

def ms_to_index(ms: float, total_values: int, total_ms: float):
    return int((ms / total_ms) * total_values)

def create_splits_and_classify_from_wav(
    input_file_path: str,
    output_directory: str,
    win_size_ms: int,
    stride_ms: int,
    beam_endpoint: str,
    beam_access_key: str,
    audio_freq: float,
    values_count: int,
):
    audio = AudioSegment.from_wav(input_file_path)
    audio_len = len(audio)
    win_size_index = math.ceil(float(win_size_ms) * (audio_freq / 1000))
    stride_index = math.ceil(float(stride_ms) * (audio_freq / 1000))
    print(f", length={audio_len}ms:")
    fname = input_file_path.split("/")[-1].split(".")[:-1]
    fname = ".".join(fname)
    output_subdirectory = f"{output_directory}/{fname}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(output_subdirectory):
        os.makedirs(output_subdirectory)
    intervals_list = []
    if win_size_index == 0:
        win_size_index = values_count
    windows = []
    for i in range(0, values_count - win_size_index + 1, stride_index):
        start = i
        end = i + win_size_index
        if end > values_count:
            end = values_count
        windows.append([start, end])
    windows.append([values_count - win_size_index, values_count])
    for [start_index, end_index] in windows:
        start_ms = math.floor(float(start_index) / ((audio_freq / 1000)))
        end_ms = math.ceil(float(end_index) / ((audio_freq / 1000)))
        print('    [' + str(start_ms) + ' - ' + str(end_ms) + 'ms.] ', end='')
        split_audio = audio[start_ms:end_ms]
        fname_split = f"{fname}_{start_ms}_{end_ms}.wav"
        output_file = os.path.join(output_subdirectory, fname_split)
        split_audio.export(output_file, format="wav")
        # Call Beam.cloud endpoint
        classification = classify_audio_sample(output_file, beam_endpoint, beam_access_key, list(audioset_labels_list.keys()))
        if not isinstance(classification, list):
            print('Beam endpoint did not return a list:', classification)
            exit(1)
        if len(classification) == 0:
            print('Beam endpoint did not return any classifications:', classification)
            exit(1)
        if 'score' not in classification[0]:
            print('Beam endpoint did not return a classification with "score" in it:', classification)
            exit(1)
        top_n_labels_count = min(3, len(classification))
        top_n_labels_str = ', '.join([f'{x["label"].lower()} ({float("{:.3f}".format(x["score"]))})' for x in classification[0:top_n_labels_count]])
        print(f'top results: {top_n_labels_str}', end='')
        label = other_label
        for c in classification:
            if c['label'].lower() in audioset_labels_list.keys():
                if c['score'] >= min_confidence:
                    label = audioset_labels_list[c['label'].lower()]
                    break
        print(f': result={label}')
        multilabel_entry = (label, start_index, end_index)
        intervals_list.append(multilabel_entry)
    return intervals_list

def set_sample_label_in_studio(api: RawDataApi, project_id: int, sample_id: int, label: str):
    label_dict = {"label": label}
    set_sample_label_request = EditSampleLabelRequest.from_dict(label_dict)
    rc = api.edit_label(project_id, sample_id, set_sample_label_request)
    return rc

def append_multilabel_to_sample_in_studio(api: RawDataApi, project_id: int, sample_id: int, structured_labels: str):
    set_sample_structured_labels_request = SetSampleStructuredLabelsRequest.from_json(structured_labels)
    rc = api.set_sample_structured_labels(project_id, sample_id, set_sample_structured_labels_request)
    return rc

def get_sample_wav_from_project_by_id(project_id, sample_id, path):
    response = raw_data_api.get_sample_as_audio(project_id=project_id, sample_id=sample_id, axis_ix=0, _preload_content=False)
    # Create the 'input' directory if it doesn't exist
    os.makedirs("input", exist_ok=True)
    # Save the file in the 'input' folder
    input_path = os.path.join("input", f"{sample.filename}.wav")
    with open(input_path, "wb") as f:
        f.write(response.data)
    return input_path

#####################
### Begin entry point
#####################
config = Configuration(host=EI_API_ENDPOINT, api_key={"ApiKeyAuthentication": EI_PROJECT_API_KEY})
client = ApiClient(config)
projects_api = ProjectsApi(client)
raw_data_api = RawDataApi(client)
project_id = None
response = projects_api.list_projects()
if not hasattr(response, "success") or getattr(response, "success") == False:
    raise RuntimeError("Could not obtain the project ID.")
else:
    project_id = response.projects[0].id
print(f"Project ID: {project_id}")

def current_ms():
    return round(time.time() * 1000)

ix = 0
for data_id in data_ids:
    ix = ix + 1
    now = current_ms()
    sample = (raw_data_api.get_sample(project_id=project_id, sample_id=data_id, proposed_actions_job_id=args.propose_actions)).sample
    prefix = '[' + str(ix).rjust(len(str(len(data_ids))), ' ') + '/' + str(len(data_ids)) + ']'
    print(prefix, 'Labeling ' + sample.filename + ' (ID ' + str(sample.id) + ')', end='')
    total_length_ms = sample.total_length_ms
    values_count = sample.values_count
    sample_file_path = os.path.join("input", f"{sample.filename}.wav")
    sample_data = get_sample_wav_from_project_by_id(project_id, sample.id, sample_file_path)
    intervals_list = create_splits_and_classify_from_wav(
        sample_file_path,
        OUTPUT_PATH,
        win_size_ms,
        win_stride_ms,
        BEAM_ENDPOINT,
        BEAM_ACCESS_KEY,
        sample.frequency,
        values_count
    )
    max_ix = values_count - 1
    structured_labels = []
    last_index = 0
    for interval in intervals_list:
        label, start_ix, end_ix = interval
        if end_ix > max_ix:
            end_ix = max_ix
        structured_labels.append({
            'startIndex': start_ix,
            'endIndex': end_ix,
            'label': label
        })
        last_index = end_ix + 1
    new_metadata = sample.metadata if sample.metadata else {}
    new_metadata['labeled_by'] = 'audio-spectrogram-transformer-beam'
    new_metadata['ast_labels'] = audioset_labels_list_str
    if args.propose_actions:
        raw_data_api.set_sample_proposed_changes(project_id=project_id, sample_id=sample.id, set_sample_proposed_changes_request={
            'jobId': args.propose_actions,
            'proposedChanges': {
                'structuredLabels': structured_labels,
                'metadata': new_metadata
            }
        })
    else:
        raw_data_api.set_sample_structured_labels(
            project_id, sample.id, set_sample_structured_labels_request={
                'structuredLabels': structured_labels
            }
        )
        raw_data_api.set_sample_metadata(project_id=project_id, sample_id=sample.id, set_sample_metadata_request={
            'metadata': new_metadata
        })
print('All done!')
