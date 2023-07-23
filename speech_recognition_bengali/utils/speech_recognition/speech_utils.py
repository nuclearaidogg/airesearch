from google.cloud import storage
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess
import librosa
import numpy as np
import io
from google.oauth2 import service_account
from google.cloud import bigquery
import zipfile

def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def run_shell_command(command):

    import subprocess
    # Use subprocess.run() to run the command
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Get the output and error messages (if any)
        output = result.stdout
        error = result.stderr
        
        # Print the output and error messages
        # print("Output:")
        # print(output)
        
        # print("\nError:")
        # print(error)

    except subprocess.CalledProcessError as e:
        # This exception is raised if the command returns a non-zero exit code
        print("Error occurred:", e)


def download_data_from_kaggle(kaggle_api,destination):

    create_folder_if_not_exist(destination)
    # run_shell_command(f"cd {destination}")
    run_shell_command(f"{kaggle_api} -p {destination}")
    print(f"Downloaded the data in the location {destination}")

    # Replace 'ls' with the terminal command you want to execute


def connect_to_gcs_bucket(bucket_name):
    # Initialize the GCS client
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/danger/airesearch/secret/lucky-starlight-391909-699ae242ee70.json'
    
    client = storage.Client()
    print(client)
    # Get the bucket reference
    bucket = client.get_bucket(bucket_name)

    return bucket


def upload_folder_to_bucket(bucket_name, source_folder_path, destination_folder_path=""):
    # Create a storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Upload files from the folder to the bucket
    for root, dirs, files in os.walk(source_folder_path):
        for file in files:
            source_file_path = os.path.join(root, file)
            destination_blob_name = os.path.join(destination_folder_path, file)
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_path)

def upload_blob(project_id,bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the GCS bucket."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def stream_files_from_gcp_bucket(bucket,prefix,batch_size):
    # Use a page_iterator to paginate through the objects in batches
    blobs = bucket.list_blobs(prefix=prefix)
    batch = []
    for blob in blobs:
        batch.append(blob)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    yield batch


def mp3_to_vectors(mp3_files_batch):
    vectors_batch = []
    ids = []
    for mp3_file_blob in mp3_files_batch:
        # Download the MP3 file content as bytes
        mp3_data = mp3_file_blob.download_as_bytes()

        file_name = mp3_file_blob.name

        # Convert MP3 bytes to a numpy array (samples) and sample rate
        samples, sample_rate = librosa.load(io.BytesIO(mp3_data), sr=None)

        # Perform audio processing or feature extraction (here, we're using a simple example)
        # For example, we can extract the Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate)

        # Append the MFCCs to the vectors_batch
        vectors_batch.append(mfccs)
        ids.append(file_name)

    return vectors_batch,ids

def unzip_folder_on_bucket(bucket_name, zip_file_path, destination_folder,from_bucket = False):
    # Set up Google Cloud Storage client using service account credentials
    credentials = service_account.Credentials.from_service_account_file('/Users/danger/airesearch/secret/lucky-starlight-391909-699ae242ee70.json')
    storage_client = storage.Client(credentials=credentials)

    # Get the bucket and blob (zip file)
    bucket = storage_client.bucket(bucket_name)


    if from_bucket == True:
        blob = bucket.blob(zip_file_path)
        # Download the zip file from the bucket to memory
        zip_data = io.BytesIO()
        blob.download_to_file(zip_data)
        zip_data.seek(0)  # Reset the buffer's position
    else:
        # Read the local ZIP file into memory
        with open(zip_file_path, 'rb') as zip_file:
            zip_data = io.BytesIO(zip_file.read())

    # Unzip the data directly to the destination folder on the bucket
    with zipfile.ZipFile(zip_data, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            file_path = file_info.filename
            # Concatenate the destination folder and file path to create the blob name
            blob_name = f"{destination_folder}/{file_path}"
            file_data = zip_ref.read(file_info.filename)
            # Upload the unzipped file to the bucket
            bucket.blob(blob_name).upload_from_string(file_data)

def update_bigquery_table_if_exist(project_id,dataset_id,table_id,schema,folder_name,batch_size,bucket):

    # Create a BigQuery client
    client = bigquery.Client(project=project_id)

    # Define the schema of the BigQuery table (modify this to match your data)
    # schema = [
    #     bigquery.SchemaField("file_name", "STRING"),
    #     bigquery.SchemaField("vector_data", "FLOAT64", mode="REPEATED"),  # Assuming the vectors are float values
    # ]

    # Get a reference to the BigQuery table
    table_ref = bigquery.TableReference.from_string(f"{project_id}.{dataset_id}.{table_id}")

    # Check if the table exists
    try:
        table = client.get_table(table_ref)
    except Exception:
        # If the table doesn't exist, create it
        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table)


    for mp3_files_batch in stream_files_from_gcp_bucket(bucket, folder_name, batch_size):
        vectors_with_names_batch = mp3_to_vectors(mp3_files_batch)

        # Convert the vectors_with_names_batch to a list of dictionaries for BigQuery insertion
        rows_to_insert = [{"file_name": file_name, "vector_data": vector_data.tolist()} for file_name, vector_data in vectors_with_names_batch]

        # Insert rows into the BigQuery table in batches (you can set the batch size as needed)
        for i in range(0, len(rows_to_insert), batch_size):
            client.insert_rows_json(table, rows_to_insert[i:i + batch_size])

def stream_data_from_bigquery(project_id,dataset_id,table_id,batch_size,features,target):

    from google.cloud import bigquery

    # Create a BigQuery client
    client = bigquery.Client(project=project_id)

    # Get a reference to the BigQuery table
    table_ref = bigquery.TableReference.from_string(f"{project_id}.{dataset_id}.{table_id}")

    # Define a query to retrieve data from the table
    query = f"SELECT * FROM `{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}`"

    # Create a query job
    job = client.query(query)

    # Fetch data in batches using pagination
    while True:
        rows = job.result(page_size=batch_size).to_dataframe()
        if rows.empty:
            # No more rows to fetch, exit the loop
            break
        yield rows[features].to_numpy(),rows[target].to_numpy()


def stream_data_from_bigquery(project_id,dataset_id,table_id,batch_size,features,target,num_streams,batch_in_thread = True,row_restriction = ''):

    from tensorflow.python.framework import dtypes
    from tensorflow_io.bigquery import BigQueryClient
    import tensorflow as tf

    tensorflow_io_bigquery_client = BigQueryClient()

    if type(target) == list:
        t = target
    else:
        t = [target]

    training_features = []
    for feature in features:
        if type(feature) == list:
            training_features+=feature
        else:
            train_features+=[feature]
    
    selected_fields = train_features + target
    read_session = tensorflow_io_bigquery_client.read_session(
                                                                parent = "projects/"+project_id,
                                                                project_id = project_id,
                                                                table_id = table_id,
                                                                dataset_id = dataset_id,
                                                                selected_fields = selected_fields,
                                                                output_types = [dtypes.float64 for i in selected_fields],
                                                                row_restriction = row_restriction,
                                                                requested_streams = num_streams,
                                                                data_format = BigQueryClient.DataFormat.AVRO
                                                            )

    streams = read_session.get_streams()
    print("BigQuery returned {} streams".format(len(streams)))

    streams_count = tf.size(streams)
    streams_count64 = tf.cast(streams_count, dtype = tf.int64)

    def _preprocess(row_batch):
        if len(t) == 1:
            labels = row_batch[t[0]]
        else:
            labels = tf.stack([row_batch[key] for key in t], axis = 1)
        
        feature_list = []
        for feature in features:
            temp = tf.stack([row_batch[key] for key in feature],axis = 1)
            feature_list.append(temp)

        return tuple(feature_list) , labels

    def _read_rows(stream):

        dataset = read_session.read_rows(streams)

        if(batch_in_thread):
            dataset = dataset.batch(batch_size)

        return dataset

    stream_ds = tf.data.Dataset.from_tensor_slices(streams)

    dataset = stream_ds.interleave(
                                    _read_rows,
                                    cycle_length = streams_count64,
                                    num_paraller_calls = streams_count64,
                                    deterministic = False
                                )

    if(not batch_in_thread):
        dataset = dataset.batch(batch_size)

    dataset = dataset.map(_preprocess,num_parallel_calls = tf.data.experimental.AUTOTUNE , deterministic = False)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


if __name__=="__main__":

    # Authenticate using the service account key JSON file

    run_shell_command('cd /Users/danger/airesearch/airesearch/speech_recognition_bengali/')
    run_shell_command('export GOOGLE_APPLICATION_CREDENTIALS=/Users/danger/airesearch/secret/lucky-starlight-391909-699ae242ee70.json')
    run_shell_command('echo $GOOGLE_APPLICATION_CREDENTIALS')
    run_shell_command('export KAGGLE_USERNAME=dangerwhite')
    run_shell_command('export KAGGLE_KEY=/users/danger/.kaggle/kaggle.json')
    
    bucket_name = "nuclearairesearch"
    bucket = connect_to_gcs_bucket(bucket_name)
    prefix = "/bengaliai/train_mp3s/"
    batch_size = 10
    print(bucket)
    project_id="lucky-starlight-391909"
    dataset_id="speech_research"
    table_id="Speech_mp3_vectors"
    schema= [bigquery.SchemaField("file_name", "STRING"),
                bigquery.SchemaField("vector_data", "FLOAT64", mode="REPEATED")]
    folder_name="bengaliai/train_mp3s"
    print("Downloading data from kaggle...")

    # download_data_from_kaggle('kaggle competitions download -c bengaliai-speech','/users/danger/temp/')

    # run_shell_command('unzip  /Users/danger/airesearch/airesearch/speech_recognition_bengali/bengaliai-speech.zip -d ./input')

    zip_blob_name = '/Users/danger/airesearch/airesearch/speech_recognition_bengali/bengaliai-speech.zip'

    # unzip_folder_on_bucket(bucket_name, zip_blob_name, "bengaliai/inputs/")

    # upload_folder_to_bucket(bucket_name, '/users/danger/temp/', destination_folder_path="bengaliai/inputs/")

    update_bigquery_table_if_exist(project_id,dataset_id,table_id,schema,folder_name,batch_size,bucket)

