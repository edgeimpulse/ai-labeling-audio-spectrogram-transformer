import os, sys
import requests
import time
import argparse
import json
import math

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
if not os.getenv("HF_API_KEY"):
    print("Missing HF_API_KEY")
    sys.exit(1)

## Property of the project
if not os.getenv("EI_PROJECT_API_KEY"):
    print("Missing EI_PROJECT_API_KEY")
    sys.exit(1)

HF_API_KEY = os.environ.get("HF_API_KEY")

EI_PROJECT_API_KEY = os.environ.get("EI_PROJECT_API_KEY")
EI_API_ENDPOINT = os.environ.get("EI_API_ENDPOINT", "https://studio.edgeimpulse.com/v1")

# Inferencing of audio classifier
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models/MIT/ast-finetuned-audioset-10-10-0.4593"

# Settings
OUTPUT_PATH = "./out"

# Argument parser for command line arguments

parser = argparse.ArgumentParser(
    description="Use Hugging Face to classify sound sample in your dataset"
)
# parser.add_argument(
#     "--in-file",
#     type=str,
#     required=False,
#     help="Argument passed by Edge Impulse transformation block when the --in-file option is selected",
# )
parser.add_argument(
    "--audioset-labels",
    type=str,
    required=True,
    help='Comma separated list of labels from "AudioSet" that will be be used to label the sample. When model returns any other label, the label for this sample in Edge Impulse will be set to "other".',
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

audioset_labels_list = [ x.strip().lower() for x in args.audioset_labels.split(",") ]
win_size_ms = args.win_size_ms
win_stride_ms = args.win_stride_ms
if args.data_ids_file:
    with open(args.data_ids_file, 'r') as f:
        data_ids = json.load(f)
other_label = args.other_label
min_confidence = args.min_confidence

print('Labeling data using Audio Spectrogram Transformers')
print('')
print('Detecting audio:')
print('    Audioset labels:', audioset_labels_list)
print('    Other label:', other_label)
print('    Min. confidence:', min_confidence)
print(f"    Window size: {win_size_ms}ms.")
print(f"    Window stride: {win_stride_ms}ms.")
if (len(data_ids) < 6):
    print('    IDs:', ', '.join([ str(x) for x in data_ids ]))
else:
    print('    IDs:', ', '.join([ str(x) for x in data_ids[0:5] ]), 'and ' + str(len(data_ids) - 5) + ' others')

print('')

def classify_audio_sample(filename: str, hf_api_key: str):
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(HF_INFERENCE_URL, headers=headers, data=data)
    body = response.json()

    if (type(body) is dict and 'estimated_time' in body.keys()):
        print('Request failed, model is spinning up:' + body['error'] + ', sleeping for ' + str(body['estimated_time'] + 5) + ' seconds...')
        time.sleep(body['estimated_time'] + 5)
        response = requests.post(HF_INFERENCE_URL, headers=headers, data=data)
        body = response.json()

    return body


def ms_to_index(ms: float, total_values: int, total_ms: float):
    return int((ms / total_ms) * total_values)

def create_splits_and_classify_from_wav(
    input_file_path: str,
    output_directory: str,
    win_size_ms: int,
    stride_ms: int,
    hf_api_key: str,
    audioset_labels_list: list,
    audio_freq: float,
    values_count: int,
):
    """
    Split the input audio file into windows of size "win_size_ms" with stride "stride_ms"
    Then put each split though audio classifier and rename the file with the label
    If the multilabel_output is enabled, return a structured multilabel list and files list
    """

    # Load the input audio file
    audio = AudioSegment.from_wav(input_file_path)
    audio_len = len(audio)

    win_size_index = math.ceil(float(win_size_ms) * (audio_freq / 1000))
    stride_index = math.ceil(float(stride_ms) * (audio_freq / 1000))

    print(f", length={audio_len}ms:")

    # File name all until file extension, preserve other dots in file name
    fname = input_file_path.split("/")[-1].split(".")[:-1]
    fname = ".".join(fname)

    output_subdirectory = f"{output_directory}/{fname}"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(output_subdirectory):
        os.makedirs(output_subdirectory)

    intervals_list = (
        []
    )

    # if 0 is specified we give one label per sample and send the whole sample to classify for a model
    # THis means the loop will execute only once and one interval will be added to the list
    if win_size_index == 0:
        win_size_index = values_count

    windows = []
    for i in range(0, values_count - win_size_index + 1, stride_index):
        start = i
        end = i + win_size_index
        if end > values_count:
            end = values_count
        windows.append([ start, end ])

    windows.append([ values_count - win_size_index, values_count ])

    # print('audio_len', audio_len)
    # print('values_count', values_count)
    # print('windows', windows)

    # Get sliding window of size "win_size_ms" with stride "stride_ms" from audio wav
    for [ start_index, end_index ] in windows:
        start_ms = math.floor(float(start_index) / ((audio_freq / 1000)))
        end_ms = math.ceil(float(end_index) / ((audio_freq / 1000)))
        print('    [' + str(start_ms) + ' - ' + str(end_ms) + 'ms.] ', end='')

        # Get the split audio
        split_audio = audio[start_ms:end_ms]
        # create a filename for split to identify by section
        fname_split = f"{fname}_{start_ms}_{end_ms}.wav"
        output_file = os.path.join(output_subdirectory, fname_split)

        split_audio.export(output_file, format="wav")
        # do inference
        classification = classify_audio_sample(output_file, hf_api_key)

        if (not isinstance(classification, (list))):
            print('classify_audio_sample did not return a list:', classification)
            exit(1)

        if (len(classification) == 0):
            print('classify_audio_sample did not return any classifications:', classification)
            exit(1)

        if (not 'score' in classification[0].keys()):
            print('classify_audio_sample did not return a classification with "score" in it:', classification)
            exit(1)

        top_n_labels_count = min(3, len(classification))
        top_n_labels_str = ', '.join([ f'{x["label"].lower()} ({float("{:.3f}".format(x["score"]))})' for x in classification[0:top_n_labels_count] ])
        print(f'top results: {top_n_labels_str}', end='')

        # loop through classification list (sorted by score already)
        # and find the first one in the audioset_labels_list and see if we're >= min confidence
        label = other_label
        for c in classification:
            if (c['label'].lower() in audioset_labels_list):
                if (c['score'] >= min_confidence):
                    label = c['label'].lower()
                    break

        print(f': result={label}')

        # Add label and interval tuple to split list
        multilabel_entry = (label, start_index, end_index)
        intervals_list.append(multilabel_entry)

    return intervals_list


def set_sample_label_in_studio(
    api: RawDataApi, project_id: int, sample_id: int, label: str
):
    label_dict = {"label": label}
    set_sample_label_request = EditSampleLabelRequest.from_dict(label_dict)
    rc = api.edit_label(project_id, sample_id, set_sample_label_request)

    return rc


def append_multilabel_to_sample_in_studio(
    api: RawDataApi, project_id: int, sample_id: int, structured_labels: str
):
    set_sample_structured_labels_request = SetSampleStructuredLabelsRequest.from_json(
        structured_labels
    )
    # print(set_sample_structured_labels_request)
    rc = api.set_sample_structured_labels(
        project_id, sample_id, set_sample_structured_labels_request
    )

    return rc

def get_sample_wav_from_project_by_id(project_id, sample_id, path):
    # returns string
    response = raw_data_api.get_sample_as_audio(
        project_id=project_id, sample_id=sample_id, axis_ix=0, _preload_content=False
    )
    # Save binary string as a WAV file
    with open(path, "wb") as f:
        f.write(response.data)
    # print(response)
    # return 'output.wav'


#####################
### Begin entry point
#####################

# Create top-level API client
config = Configuration(
    host=EI_API_ENDPOINT, api_key={"ApiKeyAuthentication": EI_PROJECT_API_KEY}
)
client = ApiClient(config)

# Instantiate sub-clients
projects_api = ProjectsApi(client)
raw_data_api = RawDataApi(client)

project_id = None
# Get the project ID, which we'll need for future API calls
response = (
    projects_api.list_projects()
)  # BC we have PI key it will list only this project
if not hasattr(response, "success") or getattr(response, "success") == False:
    raise RuntimeError("Could not obtain the project ID.")
else:
    project_id = response.projects[0].id

# Print the project ID
print(f"Project ID: {project_id}")

# Get relevant project info
project_info = projects_api.get_project_info(project_id=project_id).to_dict()

def current_ms():
    return round(time.time() * 1000)

ix = 0
for data_id in data_ids:
    ix = ix + 1
    now = current_ms()

    sample = (raw_data_api.get_sample(project_id=project_id, sample_id=data_id)).sample

    prefix = '[' + str(ix).rjust(len(str(len(data_ids))), ' ') + '/' + str(len(data_ids)) + ']'

    print(prefix, 'Labeling ' + sample.filename + ' (ID ' + str(sample.id) + ')', end='')

    # # Skip sample if it already has structured labels
    # if not "structuredLabels" in sample:
    #     print(f"Skipping sample {sample_id} as it already has structured labels")
    #     continue

    total_length_ms = sample.total_length_ms  # length in Ms
    values_count = sample.values_count  # length in indices

    # Get the audio file from the project (from the API)
    sample_file_path = f"./{sample.filename}.wav"
    sample_data = get_sample_wav_from_project_by_id(
        project_id, sample.id, sample_file_path
    )

    # Create splits and classify
    intervals_list = create_splits_and_classify_from_wav(
        sample_file_path,
        OUTPUT_PATH,
        win_size_ms,
        win_stride_ms,
        HF_API_KEY,
        audioset_labels_list,
        sample.frequency,
        values_count
    )

    max_ix = values_count - 1
    structured_labels = []
    last_index = 0
    for interval in intervals_list:
        label, start_ix, end_ix = interval
        if (end_ix > max_ix): end_ix = max_ix
        structured_labels.append({
            'startIndex': start_ix,
            'endIndex': end_ix,
            'label': label
        })
        last_index = end_ix + 1

    new_metadata = sample.metadata if sample.metadata else { }
    new_metadata['labeled_by'] = 'audio-spectrogram-transformer'
    new_metadata['labels'] = ', '.join(audioset_labels_list)

    if args.propose_actions:
        raw_data_api.set_sample_proposed_changes(project_id=project_id, sample_id=sample.id, set_sample_proposed_changes_request={
            'jobId': args.propose_actions,
            'proposedChanges': {
                'structuredLabels': structured_labels,
                'metadata': new_metadata
            }
        })
    else:
        # print(set_sample_structured_labels_request)
        raw_data_api.set_sample_structured_labels(
            project_id, sample.id, set_sample_structured_labels_request={
                'structuredLabels': structured_labels
            }
        )
        raw_data_api.set_sample_metadata(project_id=project_id, sample_id=sample.id, set_sample_metadata_request={
            'metadata': new_metadata
        })

print('All done!')
