RAG Engine API 

bookmark_border
The Vertex AI RAG Engine is a component of the Vertex AI platform, which facilitates Retrieval-Augmented Generation (RAG). RAG Engine enables Large Language Models (LLMs) to access and incorporate data from external knowledge sources, such as documents and databases. By using RAG, LLMs can generate more accurate and informative LLM responses.

Parameters list
This section lists the following:

Parameters	Examples
See Corpus management parameters.	See Corpus management examples.
See File management parameters.	See File management examples.
Corpus management parameters
For information about a RAG corpus, see Corpus management.

Create a RAG corpus
This table lists the parameters used to create a RAG corpus.

Body Request
Parameters
display_name

Required: string

The display name of the RAG corpus.

description

Optional: string

The description of the RAG corpus.

vector_db_config

Optional: Immutable: RagVectorDbConfig

The configuration for the Vector DBs.

vertex_ai_search_config.serving_config

Optional: string

The configuration for the Vertex AI Search.

Format: projects/{project}/locations/{location}/collections/{collection}/engines/{engine}/servingConfigs/{serving_config} or projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/servingConfigs/{serving_config}

RagVectorDbConfig
Parameters
rag_managed_db

oneof vector_db: RagVectorDbConfig.RagManagedDb

If no vector database is specified, rag_managed_db is the default vector database.

pinecone

oneof vector_db: RagVectorDbConfig.Pinecone

Specifies your Pinecone instance.

pinecone.index_name

string

This is the name used to create the Pinecone index that's used with the RAG corpus.

This value can't be changed after it's set. You can leave it empty in the CreateRagCorpus API call, and set it with a non-empty value in a follow up UpdateRagCorpus API call.

vertex_vector_search

oneof vector_db: RagVectorDbConfig.VertexVectorSearch

Specifies your Vertex Vector Search instance.

vertex_vector_search.index

string

This is the resource name of the Vector Search index that's used with the RAG corpus.

Format: projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}

This value can't be changed after it's set. You can leave it empty in the CreateRagCorpus API call, and set it with a non-empty value in a follow up UpdateRagCorpus API call.

vertex_vector_search.index_endpoint

string

This is the resource name of the Vector Search index endpoint that's used with the RAG corpus.

Format: projects/{project}/locations/{location}/indexes/{index}

This value can't be changed after it's set. You can leave it empty in the CreateRagCorpus API call, and set it with a non-empty value in a follow up UpdateRagCorpus API call.

api_auth.api_key_config.api_key_secret_version

string

This the full resource name of the secret that is stored in Secret Manager, which contains your Pinecone API key.

Format: projects/{PROJECT_NUMBER}/secrets/{SECRET_ID}/versions/{VERSION_ID}

You can leave it empty in the CreateRagCorpus API call, and set it with a non-empty value in a follow up UpdateRagCorpus API call.

rag_embedding_model_config.vertex_prediction_endpoint.endpoint

Optional: Immutable: string

The embedding model to use for the RAG corpus. This value can't be changed after it's set. If you leave it empty, we use text-embedding-005 as the embedding model.

Update a RAG corpus
This table lists the parameters used to update a RAG corpus.

Body Request
Parameters
display_name

Optional: string

The display name of the RAG corpus.

description

Optional: string

The description of the RAG corpus.

rag_vector_db.pinecone.index_name

string

This is the name used to create the Pinecone index that's used with the RAG corpus.

If your RagCorpus was created with a Pinecone configuration, and this field has never been set before, then you can update the Pinecone instance's index name.

rag_vector_db.vertex_vector_search.index

string

This is the resource name of the Vector Search index that's used with the RAG corpus.

Format: projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}

If your RagCorpus was created with a Vector Search configuration, and this field has never been set before, then you can update it.

rag_vector_db.vertex_vector_search.index_endpoint

string

This is the resource name of the Vector Search index endpoint that's used with the RAG corpus.

Format: projects/{project}/locations/{location}/indexes/{index}

If your RagCorpus was created with a Vector Search configuration, and this field has never been set before, then you can update it.

rag_vector_db.api_auth.api_key_config.api_key_secret_version

string

The full resource name of the secret that is stored in Secret Manager, which contains your Pinecone API key.

Format: projects/{PROJECT_NUMBER}/secrets/{SECRET_ID}/versions/{VERSION_ID}

List RAG corpora
This table lists the parameters used to list RAG corpora.

Parameters
page_size

Optional: int

The standard list page size.

page_token

Optional: string

The standard list page token. Typically obtained from [ListRagCorporaResponse.next_page_token][] of the previous [VertexRagDataService.ListRagCorpora][] call.

Get a RAG corpus
This table lists parameters used to get a RAG corpus.

Parameters
name

string

The name of the RagCorpus resource. Format: projects/{project}/locations/{location}/ragCorpora/{rag_corpus_id}

Delete a RAG corpus
This table lists parameters used to delete a RAG corpus.

Parameters
name

string

The name of the RagCorpus resource. Format: projects/{project}/locations/{location}/ragCorpora/{rag_corpus_id}

File management parameters
For information about a RAG file, see File management.

Upload a RAG file
This table lists parameters used to upload a RAG file.

Body Request
Parameters
parent

string

The name of the RagCorpus resource. Format: projects/{project}/locations/{location}/ragCorpora/{rag_corpus_id}

rag_file

Required: RagFile

The file to upload.

upload_rag_file_config

Required: UploadRagFileConfig

The configuration for the RagFile to be uploaded into the RagCorpus.

RagFile
display_name

Required: string

The display name of the RAG file.

description

Optional: string

The description of the RAG file.

UploadRagFileConfig
rag_file_transformation_config.rag_file_chunking_config.fixed_length_chunking.chunk_size

int32

Number of tokens each chunk has.

rag_file_transformation_config.rag_file_chunking_config.fixed_length_chunking.chunk_overlap

int32

The overlap between chunks.

Import RAG files
This table lists parameters used to import a RAG file.

Parameters
parent

Required: string

The name of the RagCorpus resource.

Format: projects/{project}/locations/{location}/ragCorpora/{rag_corpus_id}

gcs_source

oneof import_source: GcsSource

Cloud Storage location.

Supports importing individual files as well as entire Cloud Storage directories.

gcs_source.uris

list of string

Cloud Storage URI that contains the upload file.

google_drive_source

oneof import_source: GoogleDriveSource

Google Drive location.

Supports importing individual files as well as Google Drive folders.

slack_source

oneof import_source: SlackSource

The slack channel where the file is uploaded.

jira_source

oneof import_source: JiraSource

The Jira query where the file is uploaded.

share_point_sources

oneof import_source: SharePointSources

The SharePoint sources where the file is uploaded.

rag_file_transformation_config.rag_file_chunking_config.fixed_length_chunking.chunk_size

int32

Number of tokens each chunk has.

rag_file_transformation_config.rag_file_chunking_config.fixed_length_chunking.chunk_overlap

int32

The overlap between chunks.

rag_file_parsing_config

Optional: RagFileParsingConfig

Specifies the parsing configuration for RagFiles.

If this field isn't set, RAG uses the default parser.

max_embedding_requests_per_min

Optional: int32

The maximum number of queries per minute that this job is allowed to make to the embedding model specified on the corpus. This value is specific to this job and not shared across other import jobs. Consult the Quotas page on the project to set an appropriate value.

If unspecified, a default value of 1,000 QPM is used.

GoogleDriveSource
resource_ids.resource_id

Required: string

The ID of the Google Drive resource.

resource_ids.resource_type

Required: string

The type of the Google Drive resource.

SlackSource
channels.channels

Repeated: SlackSource.SlackChannels.SlackChannel

Slack channel information, include ID and time range to import.

channels.channels.channel_id

Required: string

The Slack channel ID.

channels.channels.start_time

Optional: google.protobuf.Timestamp

The starting timestamp for messages to import.

channels.channels.end_time

Optional: google.protobuf.Timestamp

The ending timestamp for messages to import.

channels.api_key_config.api_key_secret_version

Required: string

The full resource name of the secret that is stored in Secret Manager, which contains a Slack channel access token that has access to the slack channel IDs.
See: https://api.slack.com/tutorials/tracks/getting-a-token.

Format: projects/{PROJECT_NUMBER}/secrets/{SECRET_ID}/versions/{VERSION_ID}

JiraSource
jira_queries.projects

Repeated: string

A list of Jira projects to import in their entirety.

jira_queries.custom_queries

Repeated: string

A list of custom Jira queries to import. For information about JQL (Jira Query Language), see
Jira Support

jira_queries.email

Required: string

The Jira email address.

jira_queries.server_uri

Required: string

The Jira server URI.

jira_queries.api_key_config.api_key_secret_version

Required: string

The full resource name of the secret that is stored in Secret Manager, which contains Jira API key that has access to the slack channel IDs.
See: https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/

Format: projects/{PROJECT_NUMBER}/secrets/{SECRET_ID}/versions/{VERSION_ID}

SharePointSources
share_point_sources.sharepoint_folder_path

oneof in folder_source: string

The path of the SharePoint folder to download from.

share_point_sources.sharepoint_folder_id

oneof in folder_source: string

The ID of the SharePoint folder to download from.

share_point_sources.drive_name

oneof in drive_source: string

The name of the drive to download from.

share_point_sources.drive_id

oneof in drive_source: string

The ID of the drive to download from.

share_point_sources.client_id

string

The Application ID for the app registered in Microsoft Azure Portal.
The application must also be configured with MS Graph permissions "Files.ReadAll", "Sites.ReadAll" and BrowserSiteLists.Read.All.

share_point_sources.client_secret.api_key_secret_version

Required: string

The full resource name of the secret that is stored in Secret Manager, which contains the application secret for the app registered in Azure.

Format: projects/{PROJECT_NUMBER}/secrets/{SECRET_ID}/versions/{VERSION_ID}

share_point_sources.tenant_id

string

Unique identifier of the Azure Active Directory Instance.

share_point_sources.sharepoint_site_name

string

The name of the SharePoint site to download from. This can be the site name or the site id.

RagFileParsingConfig
layout_parser

oneof parser: RagFileParsingConfig.LayoutParser

The Layout Parser to use for RagFiles.

layout_parser.processor_name

string

The full resource name of a Document AI processor or processor version.

Format:
projects/{project_id}/locations/{location}/processors/{processor_id}
projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}

layout_parser.max_parsing_requests_per_min

string

The maximum number of requests the job is allowed to make to the Document AI processor per minute.

Consult https://cloud.google.com/document-ai/quotas and the Quota page for your project to set an appropriate value here. If unspecified, a default value of 120 QPM is used.

llm_parser

oneof parser: RagFileParsingConfig.LlmParser

The LLM parser to use for RagFiles.

llm_parser.model_name

string

The resource name of an LLM model.

Format:
projects/{project_id}/locations/{location}/publishers/{publisher}/models/{model}

llm_parser.max_parsing_requests_per_min

string

The maximum number of requests the job is allowed to make to the LLM model per minute.

To set an appropriate value for your project, see model quota section and the Quota page for your project to set an appropriate value here. If unspecified, a default value of 5000 QPM is used.

Get a RAG file
This table lists parameters used to get a RAG file.

Parameters
name

string

The name of the RagFile resource. Format: projects/{project}/locations/{location}/ragCorpora/{rag_file_id}

Delete a RAG file
This table lists parameters used to delete a RAG file.

Parameters
name

string

The name of the RagFile resource. Format: projects/{project}/locations/{location}/ragCorpora/{rag_file_id}

Retrieval and prediction
This section lists the retrieval and prediction parameters.

Retrieval parameters
This table lists parameters for retrieveContexts API.

Parameters
parent

Required: string

The resource name of the Location to retrieve RagContexts.
The users must have permission to make a call in the project.

Format: projects/{project}/locations/{location}

vertex_rag_store

VertexRagStore

The data source for Vertex RagStore.

query

Required: RagQuery

Single RAG retrieve query.

VertexRagStore
VertexRagStore
rag_resources

list: RagResource

The representation of the RAG source. It can be used to specify the corpus only or RagFiles. Only support one corpus or multiple files from one corpus.

rag_resources.rag_corpus

Optional: string

RagCorpora resource name.

Format: projects/{project}/locations/{location}/ragCorpora/{rag_corpus}

rag_resources.rag_file_ids

list: string

A list of RagFile resources.

Format: projects/{project}/locations/{location}/ragCorpora/{rag_corpus}/ragFiles/{rag_file}

RagQuery
text

string

The query in text format to get relevant contexts.

rag_retrieval_config

Optional: RagRetrievalConfig

The retrieval configuration for the query.

RagRetrievalConfig
top_k

Optional: int32

The number of contexts to retrieve.

filter.vector_distance_threshold

oneof vector_db_threshold: double

Only returns contexts with a vector distance smaller than the threshold.

filter.vector_similarity_threshold

oneof vector_db_threshold: double

Only returns contexts with vector similarity larger than the threshold.

ranking.rank_service.model_name

Optional: string

The model name of the rank service.

Example: semantic-ranker-512@latest

ranking.llm_ranker.model_name

Optional: string

The model name used for ranking.

Example: gemini-2.0-flash

Prediction parameters
This table lists prediction parameters.

GenerateContentRequest
tools.retrieval.vertex_rag_store

VertexRagStore

Set to use a data source powered by Vertex AI RAG store.

See VertexRagStore for details.

Corpus management examples
This section provides examples of how to use the API to manage your RAG corpus.

Create a RAG corpus example
These code samples demonstrate how to create a RAG corpus.

REST
Python

To learn how to install or update the Vertex AI SDK for Python, see Install the Vertex AI SDK for Python. For more information, see the Vertex AI SDK for Python API reference documentation.





from vertexai import rag
import vertexai

# TODO(developer): Update and un-comment below lines
# PROJECT_ID = "your-project-id"
# display_name = "test_corpus"
# description = "Corpus Description"

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

# Configure backend_config
backend_config = rag.RagVectorDbConfig(
    rag_embedding_model_config=rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model="publishers/google/models/text-embedding-005"
        )
    )
)

corpus = rag.create_corpus(
    display_name=display_name,
    description=description,
    backend_config=backend_config,
)
print(corpus)
# Example response:
# RagCorpus(name='projects/1234567890/locations/us-central1/ragCorpora/1234567890',
# display_name='test_corpus', description='Corpus Description', embedding_model_config=...
# ...
Update a RAG corpus example
You can update your RAG corpus with a new display name, description, and vector database configuration. However, you can't change the following parameters in your RAG corpus:

The vector database type. For example, you can't change the vector database from Weaviate to Vertex AI Feature Store.
If you're using the managed database option, you can't update the vector database configuration.
These examples demonstrate how to update a RAG corpus.

REST
Before using any of the request data, make the following replacements:

PROJECT_ID: Your project ID.
LOCATION: The region to process the request.
CORPUS_ID: The corpus ID of your RAG corpus.
CORPUS_DISPLAY_NAME: The display name of the RAG corpus.
CORPUS_DESCRIPTION: The description of the RAG corpus.
INDEX_NAME: The resource name of the Vector Search Index. Format: projects/{project}/locations/{location}/indexes/{index}.
INDEX_ENDPOINT_NAME: The resource name of the Vector Search index endpoint. Format: projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}.
HTTP method and URL:



PATCH https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/LOCATION/ragCorpora/CORPUS_ID
Request JSON body:



{
  "display_name" : "CORPUS_DISPLAY_NAME",
  "description": "CORPUS_DESCRIPTION",
  "rag_vector_db_config": {
    "vertex_vector_search": {
        "index": "INDEX_NAME",
        "index_endpoint": "INDEX_ENDPOINT_NAME",
    }
  }
}
To send your request, choose one of these options:

curl
Powershell
Note: The following command assumes that you have signed in to the Google Cloud CLI CLI with your user account by running gcloud CLI init or gcloud CLI auth login, or by using Cloud Shell, which automatically signs you into the gcloud CLI CLI . You can check the active account by running gcloud CLI auth list.
Save the request body in a file named request.json, and run the following command:



curl -X PATCH \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d @request.json \
    "https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/LOCATION/ragCorpora/CORPUS_ID"
You should receive a successful status code (2xx).

List RAG corpora example
These code samples demonstrate how to list all of the RAG corpora.

REST
Python

To learn how to install or update the Vertex AI SDK for Python, see Install the Vertex AI SDK for Python. For more information, see the Vertex AI SDK for Python API reference documentation.





from vertexai import rag
import vertexai

# TODO(developer): Update and un-comment below lines
# PROJECT_ID = "your-project-id"

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

corpora = rag.list_corpora()
print(corpora)
# Example response:
# ListRagCorporaPager<rag_corpora {
#   name: "projects/[PROJECT_ID]/locations/us-central1/ragCorpora/2305843009213693952"
#   display_name: "test_corpus"
#   create_time {
# ...
Get a RAG corpus example
These code samples demonstrate how to get a RAG corpus.

REST
Python

To learn how to install or update the Vertex AI SDK for Python, see Install the Vertex AI SDK for Python. For more information, see the Vertex AI SDK for Python API reference documentation.





from vertexai import rag
import vertexai

# TODO(developer): Update and un-comment below lines
# PROJECT_ID = "your-project-id"
# corpus_name = "projects/{PROJECT_ID}/locations/us-central1/ragCorpora/{rag_corpus_id}"

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

corpus = rag.get_corpus(name=corpus_name)
print(corpus)
# Example response:
# RagCorpus(name='projects/[PROJECT_ID]/locations/us-central1/ragCorpora/1234567890',
# display_name='test_corpus', description='Corpus Description',
# ...
Delete a RAG corpus example
These code samples demonstrate how to delete a RAG corpus.

REST
Python

To learn how to install or update the Vertex AI SDK for Python, see Install the Vertex AI SDK for Python. For more information, see the Vertex AI SDK for Python API reference documentation.


from vertexai import rag
import vertexai

# TODO(developer): Update and un-comment below lines
# PROJECT_ID = "your-project-id"
# corpus_name = "projects/{PROJECT_ID}/locations/us-central1/ragCorpora/{rag_corpus_id}"

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

rag.delete_corpus(name=corpus_name)
print(f"Corpus {corpus_name} deleted.")
# Example response:
# Successfully deleted the RagCorpus.
# Corpus projects/[PROJECT_ID]/locations/us-central1/ragCorpora/123456789012345 deleted.
File management examples
This section provides examples of how to use the API to manage RAG files.

Upload a RAG file example
These code samples demonstrate how to upload a RAG file.

REST
Python

To learn how to install or update the Vertex AI SDK for Python, see Install the Vertex AI SDK for Python. For more information, see the Vertex AI SDK for Python API reference documentation.


from vertexai import rag
import vertexai

# TODO(developer): Update and un-comment below lines
# PROJECT_ID = "your-project-id"
# corpus_name = "projects/{PROJECT_ID}/locations/us-central1/ragCorpora/{rag_corpus_id}"
# path = "path/to/local/file.txt"
# display_name = "file_display_name"
# description = "file description"

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

rag_file = rag.upload_file(
    corpus_name=corpus_name,
    path=path,
    display_name=display_name,
    description=description,
)
print(rag_file)
# RagFile(name='projects/[PROJECT_ID]/locations/us-central1/ragCorpora/1234567890/ragFiles/09876543',
#  display_name='file_display_name', description='file description')
Import RAG files example
Files and folders can be imported from Drive or Cloud Storage. You can use response.metadata to view partial failures, request time, and response time in the SDK's response object.

The response.skipped_rag_files_count refers to the number of files that were skipped during import. A file is skipped when the following conditions are met:

The file has already been imported.
The file hasn't changed.
The chunking configuration for the file hasn't changed.
Python
REST
from vertexai import rag
import vertexai

# TODO(developer): Update and un-comment below lines
# PROJECT_ID = "your-project-id"
# corpus_name = "projects/{PROJECT_ID}/locations/us-central1/ragCorpora/{rag_corpus_id}"
# paths = ["https://drive.google.com/file/123", "gs://my_bucket/my_files_dir"]  # Supports Cloud Storage and Google Drive Links

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

response = rag.import_files(
    corpus_name=corpus_name,
    paths=paths,
    transformation_config=rag.TransformationConfig(
        rag.ChunkingConfig(chunk_size=1024, chunk_overlap=256)
    ),
    import_result_sink="gs://sample-existing-folder/sample_import_result_unique.ndjson",  # Optional: This must be an existing Cloud Storage bucket folder, and the filename must be unique (non-existent).
    llm_parser=rag.LlmParserConfig(
      model_name="gemini-2.5-pro-preview-05-06",
      max_parsing_requests_per_min=100,
    ),  # Optional
    max_embedding_requests_per_min=900,  # Optional
)
print(f"Imported {response.imported_rag_files_count} files.")
List RAG files example
These code samples demonstrate how to list RAG files.

REST
Python

To learn how to install or update the Vertex AI SDK for Python, see Install the Vertex AI SDK for Python. For more information, see the Vertex AI SDK for Python API reference documentation.


from vertexai import rag
import vertexai

# TODO(developer): Update and un-comment below lines
# PROJECT_ID = "your-project-id"
# corpus_name = "projects/{PROJECT_ID}/locations/us-central1/ragCorpora/{rag_corpus_id}"

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

files = rag.list_files(corpus_name=corpus_name)
for file in files:
    print(file.display_name)
    print(file.name)
# Example response:
# g-drive_file.txt
# projects/1234567890/locations/us-central1/ragCorpora/111111111111/ragFiles/222222222222
# g_cloud_file.txt
# projects/1234567890/locations/us-central1/ragCorpora/111111111111/ragFiles/333333333333
Get a RAG file example
These code samples demonstrate how to get a RAG file.

REST
Python

To learn how to install or update the Vertex AI SDK for Python, see Install the Vertex AI SDK for Python. For more information, see the Vertex AI SDK for Python API reference documentation.


from vertexai import rag
import vertexai

# TODO(developer): Update and un-comment below lines
# PROJECT_ID = "your-project-id"
# file_name = "projects/{PROJECT_ID}/locations/us-central1/ragCorpora/{rag_corpus_id}/ragFiles/{rag_file_id}"

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

rag_file = rag.get_file(name=file_name)
print(rag_file)
# Example response:
# RagFile(name='projects/1234567890/locations/us-central1/ragCorpora/11111111111/ragFiles/22222222222',
# display_name='file_display_name', description='file description')
Delete a RAG file example
These code samples demonstrate how to delete a RAG file.

REST
Python

To learn how to install or update the Vertex AI SDK for Python, see Install the Vertex AI SDK for Python. For more information, see the Vertex AI SDK for Python API reference documentation.


from vertexai import rag
import vertexai

# TODO(developer): Update and un-comment below lines
# PROJECT_ID = "your-project-id"
# file_name = "projects/{PROJECT_ID}/locations/us-central1/ragCorpora/{rag_corpus_id}/ragFiles/{rag_file_id}"

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

rag.delete_file(name=file_name)
print(f"File {file_name} deleted.")
# Example response:
# Successfully deleted the RagFile.
# File projects/1234567890/locations/us-central1/ragCorpora/1111111111/ragFiles/2222222222 deleted.
Retrieval query
When a user asks a question or provides a prompt, the retrieval component in RAG searches through its knowledge base to find information that is relevant to the query.

Python
REST

To learn how to install or update the Vertex AI SDK for Python, see Install the Vertex AI SDK for Python. For more information, see the Vertex AI SDK for Python API reference documentation.


from vertexai import rag
import vertexai

# TODO(developer): Update and un-comment below lines
# PROJECT_ID = "your-project-id"
# corpus_name = "projects/[PROJECT_ID]/locations/us-central1/ragCorpora/[rag_corpus_id]"

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(
            rag_corpus=corpus_name,
            # Optional: supply IDs from `rag.list_files()`.
            # rag_file_ids=["rag-file-1", "rag-file-2", ...],
        )
    ],
    text="Hello World!",
    rag_retrieval_config=rag.RagRetrievalConfig(
        top_k=10,
        filter=rag.utils.resources.Filter(vector_distance_threshold=0.5),
    ),
)
print(response)
# Example response:
# contexts {
#   contexts {
#     source_uri: "gs://your-bucket-name/file.txt"
#     text: "....
#   ....
Generation
The LLM generates a grounded response using the retrieved contexts.

REST
Python

To learn how to install or update the Vertex AI SDK for Python, see Install the Vertex AI SDK for Python. For more information, see the Vertex AI SDK for Python API reference documentation.


from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai

# TODO(developer): Update and un-comment below lines
# PROJECT_ID = "your-project-id"
# corpus_name = "projects/{PROJECT_ID}/locations/us-central1/ragCorpora/{rag_corpus_id}"

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=corpus_name,
                    # Optional: supply IDs from `rag.list_files()`.
                    # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                )
            ],
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=10,
                filter=rag.utils.resources.Filter(vector_distance_threshold=0.5),
            ),
        ),
    )
)

rag_model = GenerativeModel(
    model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool]
)
response = rag_model.generate_content("Why is the sky blue?")
print(response.text)
# Example response:
#   The sky appears blue due to a phenomenon called Rayleigh scattering.
#   Sunlight, which contains all colors of the rainbow, is scattered
#   by the tiny particles in the Earth's atmosphere....
#   ...
What's next
To learn more about supported generation models, see Generative AI models that support RAG.
To learn more about supported embedding models, see Embedding models.
To learn more about open models, see Open models.
To learn more about RAG Engine, see RAG Engine overview.
Was this helpful?

Send feedback
Except as otherwise noted, the content of this page is licensed under the Creative Commons Attribution 4.0 License, and code samples are licensed under the Apache 2.0 License. For details, see the Google Developers Site Policies. Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2025-06-11 UTC.

Why Google
Choosing Google Cloud
Trust and security
Modern Infrastructure Cloud
Multicloud
Global infrastructure
Customers and case studies
Analyst reports
Whitepapers
Products and pricing
See all products
See all solutions
Google Cloud for Startups
Google Cloud Marketplace
Google Cloud pricing
Contact sales
Support
Google Cloud Community
Support
Release Notes
System status
Resources
GitHub
Getting Started with Google Cloud
Google Cloud documentation
Code samples
Cloud Architecture Center
Training and Certification
Developer Center
Engage
Blog
Events
X (Twitter)
Google Cloud on YouTube
Google Cloud Tech on YouTube
Become a Partner
Google Cloud Affiliate Program
Press Corner
About Google
Privacy
Site terms
Google Cloud terms
Manage cookies
Our third decade of climate action: join us
Sign up for the Google Cloud newsletter
Subscribe

English
