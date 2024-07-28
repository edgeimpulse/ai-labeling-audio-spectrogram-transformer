import os, sys
import requests
import time
import argparse

# Import the API objects we plan to use
from edgeimpulse_api import ApiClient, Configuration, ProjectsApi, RawDataApi
from edgeimpulse_api.models.set_sample_structured_labels_request import (
    SetSampleStructuredLabelsRequest,
)
from edgeimpulse_api.models.edit_sample_label_request import (
    EditSampleLabelRequest,
)

### This is a standalone block

# Multi-label is only available for pro and enterprise;
# Ingest multi-label: https://docs.edgeimpulse.com/docs/tutorials/api-examples/ingest-multi-label-data-api
# Just add the label to the existing sample:
####
# 1. Get the audio file from the project (from the API)
# 2. create_splits_and_classify
#   2.1 If Multi-label is enabled, split, create structured labels and push back the labels to the sample
#   2.2 If Multi-label is disabled, split and upload labeled files to the project

# Multi-label is only available for pro and enterprise;
# Ingest multi-label: https://docs.edgeimpulse.com/docs/tutorials/api-examples/ingest-multi-label-data-api
# Just add the label to the existing sample: https://github.com/edgeimpulse/example-multi-label-ingestion-via-api/blob/master/update-sample.sh

# If community upload back splits with labels

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

EI_INGESTION_HOST = os.environ.get("EI_INGESTION_HOST", "edgeimpulse.com")

# HF_API_KEY = ""
# EI_PROJECT_API_KEY = ""

# Inferencing of audio classifier
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models/MIT/ast-finetuned-audioset-10-10-0.4593"

# Settings
EI_API_HOST = "https://studio.edgeimpulse.com/v1"
DATASET_PATH = "dataset/gestures"
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
    "--out-directory",
    type=str,
    required=False,
    help="Directory to save images to",
    default="output",
)
parser.add_argument(
    "--audioset-labels",
    type=str,
    required=False,
    help='Comma separated list of labels from "AudioSet" that will be be used to label the sample. When model returns any other label, the label for this sample in Edge Impulse will be set to "noise". If no labels set here all the results will be added do the model',
)
parser.add_argument(
    "--my-label",
    type=str,
    required=False,
    help="A label that should be assigned to samples that classify with the labels mentioned above",
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
    help="Size of the window for each classification",
)
parser.add_argument(
    "--maybe-enabled",
    type=bool,
    required=False,
    help='If two or more highest classes are not distinctively different, the model will return "maybe" as a label',
)

args, unknown = parser.parse_known_args()

audioset_labels_list = args.audioset_labels.split(",")
print(f"Labels to get out of the model: {audioset_labels_list}")

my_label = args.my_label
print(f"Label to assigned if not noise: {my_label}")

win_size_ms = args.win_size_ms
print(f"Window size: {win_size_ms}")

win_stride_ms = args.win_stride_ms
print(f"Window stride: {win_stride_ms}")

is_maybe_enabled = args.maybe_enabled
print(f"Maybe enabled: {is_maybe_enabled}")


def classify_audio_sample(filename: str, hf_api_key: str):
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(HF_INFERENCE_URL, headers=headers, data=data)
    return response.json()


def ms_to_index(ms: float, total_values: int, total_ms: float):
    return int((ms / total_ms) * total_values)


def create_structured_labels_from_intervals_list(
    intervals: list, total_ms: float, total_values: int
):
    # Move from Ms to Index (this is so stupid) and construct a json
    start_str = '{"structuredLabels":['
    end_str = "]}"

    for i, interval in enumerate(intervals):
        label = interval[0]
        if i == 0:
            start = 0
        else:
            # add a small value to avoid overlap and cover high frequencies
            start = ms_to_index(intervals[i - 1][2], total_values, total_ms) + 1
        end = ms_to_index(interval[2], total_values, total_ms)
        if i == len(intervals) - 1:
            end = total_values - 1
        if i != 0:
            start_str += ","

        start_str += f'{{"label":"{label}","startIndex":{start},"endIndex":{end}}}'
    return start_str + end_str


def create_splits_and_classify_from_wav(
    input_file_path: str,
    output_directory: str,
    win_size_ms: int,
    stride_ms: int,
    hf_api_key: str,
    audioset_labels_list: list = None,
    my_label: str = None,
):
    """
    Split the input audio file into windows of size "win_size_ms" with stride "stride_ms"
    Then put each split though audio classifier and rename the file with the label
    If the multilabel_output is enabled, return a structured multilabel list and files list
    """

    # Load the input audio file
    audio = AudioSegment.from_wav(input_file_path)
    audio_len = len(audio)
    print(f"Audio length: {audio_len}")
    # If win_size_ms is 0, we process the given sample as a whole
    if win_size_ms == 0:
        win_size_ms = audio_len
        stride_ms = 1

    # File name all until file extension, preserve other dots in file name
    fname = input_file_path.split("/")[-1].split(".")[:-1]
    fname = ".".join(fname)
    print(fname)

    output_subdirectory = f"{output_directory}/{fname}"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(output_subdirectory):
        os.makedirs(output_subdirectory)

    intervals_list = (
        []
    )  # list of tuples (label, start_ms, end_ms) - for constructing structured_labels.labels
    out_files_list = []  # list of output file paths - for uploading to EI

    # if 0 is specified we give one label per sample and send the whole sample to classify for a model
    # THis means the loop will execute only once and one interval will be added to the list
    if win_size_ms == 0:
        win_size_ms = audio_len
    # Get sliding window of size "win_size_ms" with stride "stride_ms" from audio wav
    for i in range(0, len(audio) - win_size_ms + 1, stride_ms):
        # Get the split audio
        split_audio = audio[i : i + win_size_ms]
        # create a filename for split to identify by section
        fname_split = f"{fname}_{i}_{i + win_size_ms}.wav"
        output_file = os.path.join(output_subdirectory, fname_split)

        split_audio.export(output_file, format="wav")
        # do inference
        classification = classify_audio_sample(output_file, hf_api_key)
        print(classification)
        label = classification[0]["label"]

        # override label if it is not in the list
        if "None" not in audioset_labels_list and "none" not in audioset_labels_list:
            # # Add second chance for the label
            # if label not in audioset_labels_list and label in audioset_labels_list:
            #     label = classification[1]["label"]
            if label not in audioset_labels_list:
                label = "noise"

        # Assign the desired label to classifications from the list
        if label != "noise" and my_label != "none" and my_label != "None":
            label = my_label

        # Rename, appending label as first token
        fname_split_labeled = f"{label}.{fname}_{i}_{i + win_size_ms}.wav"
        output_file_labeled = os.path.join(output_subdirectory, fname_split_labeled)
        os.rename(output_file, output_file_labeled)

        # Add new file to list of labeled splits
        out_files_list.append(output_file_labeled)

        # Add label and interval tuple to split list
        multilabel_entry = (label, i, i + win_size_ms)
        intervals_list.append(multilabel_entry)

    return intervals_list, out_files_list


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
    print(structured_labels)
    set_sample_structured_labels_request = SetSampleStructuredLabelsRequest.from_json(
        structured_labels
    )
    # print(set_sample_structured_labels_request)
    rc = api.set_sample_structured_labels(
        project_id, sample_id, set_sample_structured_labels_request
    )

    return rc


def upload_files_to_ei_project(project_api_key, path, subset):
    """
    Upload files in the given path/subset (where subset is "training" or
    "testing")
    This is used only if splitting is performed for long samples in projects
    where the multilabel is disabled
    """

    # Construct request
    url = f"https://ingestion.edgeimpulse.com/api/{subset}/files"
    headers = {
        "x-api-key": project_api_key,
        "x-disallow-duplicates": "true",
    }

    # Get file handles and create dataset to upload
    # File names shoudl start wil {label}.filename.wav
    files = []
    file_list = os.listdir(os.path.join(path, subset))
    for file_name in file_list:
        file_path = os.path.join(path, subset, file_name)
        if os.path.isfile(file_path):
            file_handle = open(file_path, "rb")
            files.append(("data", (file_name, file_handle, "multipart/form-data")))

    # Upload the files
    response = requests.post(
        url=url,
        headers=headers,
        files=files,
    )

    # Print any errors for files that did not upload
    upload_responses = response.json()["files"]
    for resp in upload_responses:
        if not resp["success"]:
            print(resp)

    # Close all the handles
    for handle in files:
        handle[1][1].close()


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
    host=EI_API_HOST, api_key={"ApiKeyAuthentication": EI_PROJECT_API_KEY}
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
# "single_label" or "multi_label"
project_labeling_method = project_info["project"]["labelingMethod"]
# "Image classification" or "Keyword Spotting" or "Other"
project_category = project_info["project"]["category"]

print(project_category)
print(project_labeling_method)

# Get samples from train and test data of the project in one list
response = raw_data_api.list_samples(project_id=project_id, category="training")
samples = response.to_dict()["samples"]
response = raw_data_api.list_samples(project_id=project_id, category="testing")
samples.append(response.to_dict()["samples"])

for sample in samples:
    if "id" not in sample:
        continue
    sample_id = sample["id"]
    sample_filename = sample["filename"]

    # # Skip sample if it already has structured labels
    # if not "structuredLabels" in sample:
    #     print(f"Skipping sample {sample_id} as it already has structured labels")
    #     continue

    total_length_ms = sample["totalLengthMs"]  # length in Ms
    values_count = sample["valuesCount"]  # length in indices

    # Get the audio file from the project (from the API)
    sample_file_path = f"./{sample_filename}.wav"
    sample_data = get_sample_wav_from_project_by_id(
        project_id, sample_id, sample_file_path
    )

    # Create splits and classify
    intervals_list, files_list = create_splits_and_classify_from_wav(
        sample_file_path,
        OUTPUT_PATH,
        win_size_ms,
        win_stride_ms,
        HF_API_KEY,
        audioset_labels_list,
        my_label
    )

    # If the sample is short enough, just set the label
    if len(intervals_list) == 1:
        label = intervals_list[0][0]
        set_sample_label_in_studio(raw_data_api, project_id, sample_id, label)
    else:
        # Create structuredLabels.labels and append to sample
        structured_labels = create_structured_labels_from_intervals_list(
            intervals_list, total_length_ms, values_count
        )
        append_multilabel_to_sample_in_studio(
            raw_data_api, project_id, sample_id, structured_labels
        )
