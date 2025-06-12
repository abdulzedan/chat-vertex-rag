================================================
FILE: gemini/rag-engine/intro_rag_engine.ipynb
================================================
# Jupyter notebook converted to Python script.

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Intro to Building a Scalable and Modular RAG System with RAG Engine in Vertex AI 

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/intro_rag_engine.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Frag-engine%2Fintro_rag_engine.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/rag-engine/intro_rag_engine.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/intro_rag_engine.ipynb">
      <img width="32px" src="https://www.svgrepo.com/download/217753/github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

<div style="clear: both;"></div>

<b>Share to:</b>

<a href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/intro_rag_engine.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" alt="LinkedIn logo">
</a>

<a href="https://bsky.app/intent/compose?text=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/intro_rag_engine.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/7/7a/Bluesky_Logo.svg" alt="Bluesky logo">
</a>

<a href="https://twitter.com/intent/tweet?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/intro_rag_engine.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/5a/X_icon_2.svg" alt="X logo">
</a>

<a href="https://reddit.com/submit?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/intro_rag_engine.ipynb" target="_blank">
  <img width="20px" src="https://redditinc.com/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" alt="Reddit logo">
</a>

<a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/intro_rag_engine.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook logo">
</a>            
"""

"""
| | |
|-|-|
| Author(s) | [Holt Skinner](https://github.com/holtskinner) |
"""

"""
## Overview

Retrieval Augmented Generation (RAG) improves Large Language Models (LLMs) by allowing them to access and process external information sources during generation. This ensures the model's responses are grounded in factual data and avoids hallucinations.

A common problem with LLMs is that they don't understand private knowledge, that
is, your organization's data. With RAG Engine, you can enrich the
LLM context with additional private information, because the model can reduce
hallucinations and answer questions more accurately.

By combining additional knowledge sources with the existing knowledge that LLMs
have, a better context is provided. The improved context along with the query
enhances the quality of the LLM's response.

The following concepts are key to understanding Vertex AI RAG Engine. These concepts are listed in the order of the
retrieval-augmented generation (RAG) process.

1. **Data ingestion**: Intake data from different data sources. For example,
  local files, Google Cloud Storage, and Google Drive.

1. **Data transformation**: Conversion of the data in preparation for indexing. For example, data is split into chunks.

1. **Embedding**: Numerical representations of words or pieces of text. These numbers capture the
   semantic meaning and context of the text. Similar or related words or text
   tend to have similar embeddings, which means they are closer together in the
   high-dimensional vector space.

1. **Data indexing**: RAG Engine creates an index called a corpus.
   The index structures the knowledge base so it's optimized for searching. For
   example, the index is like a detailed table of contents for a massive
   reference book.

1. **Retrieval**: When a user asks a question or provides a prompt, the retrieval
  component in RAG Engine searches through its knowledge
  base to find information that is relevant to the query.

1. **Generation**: The retrieved information becomes the context added to the
  original user query as a guide for the generative AI model to generate
  factually grounded and relevant responses.

For more information, refer to the public documentation for [Vertex AI RAG Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview).
"""

"""
## Get started
"""

"""
### Install Vertex AI SDK and Google Gen AI SDK

"""

%pip install --upgrade --quiet google-cloud-aiplatform google-genai

"""
### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.
"""

import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)

"""
<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>

"""

"""
### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.
"""

import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()

"""
### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).
"""

# Use the environment variable if the user doesn't provide Project ID.
import os

from google import genai
import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

"""
### Import libraries
"""

from IPython.display import Markdown, display
from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore
from vertexai import rag

"""
### Create a RAG Corpus
"""

# Currently supports Google first-party embedding models
EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"  # @param {type:"string", isTemplate: true}

rag_corpus = rag.create_corpus(
    display_name="my-rag-corpus",
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model=EMBEDDING_MODEL
            )
        )
    ),
)

"""
### Check the corpus just created
"""

rag.list_corpora()

"""
### Upload a local file to the corpus
"""

%%writefile test.md

Retrieval-Augmented Generation (RAG) is a technique that enhances the capabilities of large language models (LLMs) by allowing them to access and incorporate external data sources when generating responses. Here's a breakdown:

**What it is:**

* **Combining Retrieval and Generation:**
    * RAG combines the strengths of information retrieval systems (like search engines) with the generative power of LLMs.
    * It enables LLMs to go beyond their pre-trained data and access up-to-date and specific information.
* **How it works:**
    * When a user asks a question, the RAG system first retrieves relevant information from external data sources (e.g., databases, documents, web pages).
    * This retrieved information is then provided to the LLM as additional context.
    * The LLM uses this augmented context to generate a more accurate and informative response.

**Why it's helpful:**

* **Access to Up-to-Date Information:**
    * LLMs are trained on static datasets, so their knowledge can become outdated. RAG allows them to access real-time or frequently updated information.
* **Improved Accuracy and Factual Grounding:**
    * RAG reduces the risk of LLM "hallucinations" (generating false or misleading information) by grounding responses in verified external data.
* **Enhanced Contextual Relevance:**
    * By providing relevant context, RAG enables LLMs to generate more precise and tailored responses to specific queries.
* **Increased Trust and Transparency:**
    * RAG can provide source citations, allowing users to verify the information and increasing trust in the LLM's responses.
* **Cost Efficiency:**
    * Rather than constantly retraining large language models, RAG allows for the introduction of new data in a more cost effective way.

In essence, RAG bridges the gap between the vast knowledge of LLMs and the need for accurate, current, and contextually relevant information.


rag_file = rag.upload_file(
    corpus_name=rag_corpus.name,
    path="test.md",
    display_name="test.md",
    description="my test file",
)

"""
### Import files from Google Cloud Storage

Remember to grant "Viewer" access to the "Vertex RAG Data Service Agent" (with the format of `service-{project_number}@gcp-sa-vertex-rag.iam.gserviceaccount.com`) for your Google Cloud Storage bucket.

For this example, we'll use a public GCS bucket containing earning reports from Alphabet.
"""

INPUT_GCS_BUCKET = (
    "gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs/"
)

response = rag.import_files(
    corpus_name=rag_corpus.name,
    paths=[INPUT_GCS_BUCKET],
    # Optional
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(chunk_size=1024, chunk_overlap=100)
    ),
    max_embedding_requests_per_min=900,  # Optional
)

"""
### Import files from Google Drive

Eligible paths can be formatted as:

- `https://drive.google.com/drive/folders/{folder_id}`
- `https://drive.google.com/file/d/{file_id}`.

Remember to grant "Viewer" access to the "Vertex RAG Data Service Agent" (with the format of `service-{project_number}@gcp-sa-vertex-rag.iam.gserviceaccount.com`) for your Drive folder/files.

"""

response = rag.import_files(
    corpus_name=rag_corpus.name,
    paths=["https://drive.google.com/drive/folders/{folder_id}"],
    # Optional
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=50)
    ),
)

"""
### Optional: Perform direct context retrieval
"""

# Direct context retrieval
response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(
            rag_corpus=rag_corpus.name,
            # Optional: supply IDs from `rag.list_files()`.
            # rag_file_ids=["rag-file-1", "rag-file-2", ...],
        )
    ],
    rag_retrieval_config=rag.RagRetrievalConfig(
        top_k=10,  # Optional
        filter=rag.Filter(
            vector_distance_threshold=0.5,  # Optional
        ),
    ),
    text="What is RAG and why it is helpful?",
)
print(response)

# Optional: The retrieved context can be passed to any SDK or model generation API to generate final results.
# context = " ".join([context.text for context in response.contexts.contexts]).replace("\n", "")

"""
### Create RAG Retrieval Tool
"""

# Create a tool for the RAG Corpus
rag_retrieval_tool = Tool(
    retrieval=Retrieval(
        vertex_rag_store=VertexRagStore(
            rag_corpora=[rag_corpus.name],
            similarity_top_k=10,
            vector_distance_threshold=0.5,
        )
    )
)

"""
### Generate Content with Gemini using RAG Retrieval Tool
"""

MODEL_ID = "gemini-2.0-flash-001"

response = client.models.generate_content(
    model=MODEL_ID,
    contents="What is RAG?",
    config=GenerateContentConfig(tools=[rag_retrieval_tool]),
)

display(Markdown(response.text))

"""
### Generate Content with Llama3 using RAG Retrieval Tool
"""

from vertexai import generative_models

# Load tool into Llama model
rag_retrieval_tool = generative_models.Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=rag_corpus.name)],
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=10,  # Optional
                filter=rag.Filter(
                    vector_distance_threshold=0.5,  # Optional
                ),
            ),
        ),
    )
)

llama_model = generative_models.GenerativeModel(
    # your self-deployed endpoint for Llama3
    "projects/{project}/locations/{location}/endpoints/{endpoint_resource_id}",
    tools=[rag_retrieval_tool],
)

response = llama_model.generate_content("What is RAG?")

display(Markdown(response.text))



================================================
FILE: gemini/rag-engine/rag_engine_eval_service_sdk.ipynb
================================================
# Jupyter notebook converted to Python script.

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
#  Evaluating Vertex RAG Engine Generation with Vertex AI Python SDK for Gen AI Evaluation Service

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_eval_service_sdk.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Frag-engine%2Frag_engine_eval_service_sdk.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/rag-engine/rag_engine_eval_service_sdk.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_eval_service_sdk.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

<div style="clear: both;"></div>

<b>Share to:</b>

<a href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_eval_service_sdk.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" alt="LinkedIn logo">
</a>

<a href="https://bsky.app/intent/compose?text=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_eval_service_sdk.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/7/7a/Bluesky_Logo.svg" alt="Bluesky logo">
</a>

<a href="https://twitter.com/intent/tweet?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_eval_service_sdk.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/53/X_logo_2023_original.svg" alt="X logo">
</a>

<a href="https://reddit.com/submit?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_eval_service_sdk.ipynb" target="_blank">
  <img width="20px" src="https://redditinc.com/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" alt="Reddit logo">
</a>

<a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_eval_service_sdk.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook logo">
</a>
"""

"""
| | |
|-|-|
| Author(s) | [Noa Ben-Efraim](https://github.com/noabenefraim/) |
"""

"""
## Overview

This notebook demonstrates how to evaluate the performance of a Retrieval Augmented Generation (RAG) engine built with Vertex AI using the Vertex AI Python SDK for Gen AI Evaluation Service. By focusing on a practical example using "Alice in Wonderland" as our knowledge base, we'll walk through the process of creating an evaluation dataset and applying custom metrics to assess the quality of generated responses.

Specifically, this notebook will guide you through:

* **Setting up a RAG Corpus:** Creating and populating a RAG corpus with a PDF document.
* **Generating Grounded Responses:** Using the Vertex AI Gemini model to produce responses based on retrieved contexts.
* **Creating an Evaluation Dataset:** Constructing a dataset with prompts, retrieved contexts, and generated responses.
* **Defining Custom Evaluation Metrics:** Implementing a custom metric to assess the accuracy, completeness, and groundedness of the generated responses.
* **Running Evaluation Tasks:** Utilizing the Vertex AI Gen AI Evaluation Service to evaluate the RAG engine's performance.
* **Analyzing Evaluation Results:** Visualizing and interpreting the evaluation results using the provided SDK tools.
"""

"""
## Get started
"""

"""
### Install Google Gen AI SDK and other required packages

"""

%pip install --upgrade --quiet google-genai google-cloud-aiplatform[evaluation] vertexai

"""
### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.
"""

import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)

"""
<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. In Colab or Colab Enterprise, you might see an error message that says "Your session crashed for an unknown reason." This is expected. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>

"""

"""
### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.
"""

import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()

"""
### Set Google Cloud project information

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).
"""

# Use the environment variable if the user doesn't provide Project ID.
import os

from google import genai
import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
vertexai.init(project=PROJECT_ID, location=LOCATION)

"""
### Import libraries
"""

from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore
import pandas as pd
from tqdm import tqdm
from vertexai import rag
from vertexai.evaluation import (
    EvalTask,
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
    notebook_utils,
)

"""
### Load model
"""

MODEL_ID = "gemini-2.0-flash-001"  # @param {type:"string"}

"""
### Create `RAGCorpus`
"""

# Currently supports Google first-party embedding models
EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"  # @param {type:"string", isTemplate: true}

rag_corpus = rag.create_corpus(
    display_name="rag-eval-corpus",
    description="A test corpus for generation evaluation",
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model=EMBEDDING_MODEL
            )
        )
    ),
)

# Get the rag corpus you just created
rag.get_corpus(rag_corpus.name)

"""
### Import files from Google Cloud Storage into `RagCorpus` (configure chunk size, chunk overlap etc as desired)

For this step you will need to create a GCS bucket, and then copy over the data from the public GCS bucket. Remember to grant "Viewer" access to the "Vertex RAG Data Service Agent" (with the format of service-{project_number}@gcp-sa-vertex-rag.iam.gserviceaccount.com) for your Google Cloud Storage bucket.

For this example, we'll use a dataset that comprises the full texts of five classic children's literature books: "The Wizard of Oz," "Gulliver's Travels," "Peter Pan," "Alice's Adventures in Wonderland," and "Through the Looking-Glass." This collection provides a rich corpus for exploring themes, characters, and settings across these iconic stories.

"""

"""
##### Copy data from public GCS bucket
"""

CURRENT_BUCKET_PATH = "gs://"  # @param {type:"string"},

PUBLIC_DATA_PATH = (
    "gs://github-repo/generative-ai/gemini/rag-engine/rag_engine_eval_service/"
)

!gsutil -m rsync -r -d $PUBLIC_DATA_PATH $CURRENT_BUCKET_PATH

"""
##### Import dataset into `RagCorpus`
"""

transformation_config = rag.TransformationConfig(
    chunking_config=rag.ChunkingConfig(
        chunk_size=512,
        chunk_overlap=100,
    ),
)

rag.import_files(
    corpus_name=rag_corpus.name,
    paths=[CURRENT_BUCKET_PATH],
    transformation_config=transformation_config,  # Optional
)

# List the files in the rag corpus
rag.list_files(rag_corpus.name)

"""
### Create RAG Retrieval Tool
"""

# Create a tool for the RAG Corpus
rag_retrieval_tool = Tool(
    retrieval=Retrieval(
        vertex_rag_store=VertexRagStore(
            rag_corpora=[rag_corpus.name],
            similarity_top_k=10,
            vector_distance_threshold=0.5,
        )
    )
)

def get_generated_response(prompt: str) -> str:
    """
    Generates a grounded response using a language model and retrieved context.

    Args:
        prompt: The input prompt for the language model.

    Returns:
        The generated text response.
    """
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=GenerateContentConfig(tools=[rag_retrieval_tool]),
    )

    return response.text

def get_retrieved_contexts(prompt: str) -> str:
    """
    Retrieves relevant contexts based on a given prompt using a RAG system.

    Args:
        prompt: The input prompt for context retrieval.

    Returns:
        A concatenated string of retrieved context texts, with newlines removed.
    """

    rag_filter = rag.utils.resources.Filter(vector_distance_threshold=0.5)

    retrieval_config = rag.RagRetrievalConfig(top_k=5, filter=rag_filter)

    response = rag.retrieval_query(
        rag_resources=[
            rag.RagResource(
                rag_corpus=rag_corpus.name,
                # Optional: supply IDs from `rag.list_files()`.
                # rag_file_ids=["rag-file-1", "rag-file-2", ...],
            )
        ],
        text=prompt,
        rag_retrieval_config=retrieval_config,
    )
    context = " ".join(
        [context.text for context in response.contexts.contexts]
    ).replace("\n", "")
    return context

"""
### Create Evaluation Dataset

Now we are prepared to create the evaluation dataset. The dataset will include:

+ Prompt: What the user is asking the RAG engine. The prompts will be a mix of inter-document and intra-document analysis.
+ Retrieved Context: The top k retrieved context from Vertex RAG Engine
+ Generated Response: The LLM generated responses grounded in the retrieved context.
"""

prompts = [
    "Compare and contrast the behaviors of the Mad Hatter and the March Hare during the tea party.",
    "What happened during Alice's croquet game with the Queen of Hearts?",
    "How did the Mad Hatter and March Hare act at the tea party?",
    "What was special about the cakes Alice ate?",
    "What happened when Gulliver first arrived in Lilliput?",
    "What was Captain Hook's main goal in Neverland?",
]

retrieved_context = []
generated_response = []
for prompt in tqdm(prompts):
    retrieved_context.append(get_retrieved_contexts(prompt))
    generated_response.append(get_generated_response(prompt))

eval_dataset = pd.DataFrame(
    {
        "prompt": prompts,
        "retrieved_context": retrieved_context,
        "response": generated_response,
    }
)

eval_dataset

"""
## Use Gen AI Evaluation Service SDK

Before diving into the evaluation process, we've set up the necessary components: a RAG corpus containing our document, a retrieval tool, and functions to generate grounded responses and retrieve relevant contexts. We've also compiled an evaluation dataset with a set of questions, the corresponding retrieved contexts, and the model's responses.

This dataset will serve as the foundation for our evaluation. We'll now leverage the Vertex AI Gen AI Evaluation Service SDK to define and apply custom metrics, allowing us to quantitatively assess the RAG engine's performance. The Gen AI Evaluation Service provides a robust framework for creating and running evaluation tasks, enabling us to gain valuable insights into the quality of our generated responses.
"""

custom_question_answering_correctness = PointwiseMetric(
    metric="custom_question_answering_correctness",
    metric_prompt_template=PointwiseMetricPromptTemplate(
        criteria={
            "accuracy": (
                "The response provides completely accurate information, consistent with the retrieved context, with no errors or omissions."
            ),
            "completeness": (
                "The response answers all parts of the question fully, utilizing the information available in the retrieved context."
            ),
            "groundedness": (
                "The response uses only the information provided in the retrieved context and does not introduce any external information or hallucinations."
            ),
        },
        rating_rubric={
            "5": "(Very good). The answer is completely accurate, complete, concise, grounded in the retrieved context, and follows all instructions.",
            "4": "(Good). The answer is mostly accurate, complete, and grounded in the retrieved context, with minor issues in conciseness or instruction following.",
            "3": "(Ok). The answer is partially accurate and complete but may have some inaccuracies, omissions, or significant issues with conciseness, groundedness, or instruction following, based on the retrieved context.",
            "2": "(Bad). The answer contains significant inaccuracies, is largely incomplete, or fails to follow key instructions, considering the information available in the retrieved context.",
            "1": "(Very bad). The answer is completely inaccurate, irrelevant, or fails to address the question in any meaningful way, based on the retrieved context.",
        },
        input_variables=["prompt", "retrieved_context"],
    ),
)

# Display the serialized metric prompt template
print(custom_question_answering_correctness.metric_prompt_template)

"""
### Run Eval Task

The Gen AI Evaluation SDK has many useful utilities to graph, summarize, and explain the evaluation results. 
"""

# Run evaluation using the custom_text_quality metric
eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=[custom_question_answering_correctness],
    experiment="test",
)
eval_result = eval_task.evaluate()

notebook_utils.display_eval_result(eval_result=eval_result)

# Example for graphing
notebook_utils.display_radar_plot(
    eval_results_with_title=[("Question answering correctness", eval_result)],
    metrics=["custom_question_answering_correctness"],
)

# Displaying explanations for one row.
notebook_utils.display_explanations(eval_result=eval_result, num=1)

"""
## Cleaning up

Delete ExperimentRun created by the evaluation.
"""

aiplatform.ExperimentRun(
    run_name=eval_result.metadata["experiment_run"],
    experiment=eval_result.metadata["experiment"],
).delete()

rag.delete_corpus(rag_corpus.name)



================================================
FILE: gemini/rag-engine/rag_engine_evaluation.ipynb
================================================
# Jupyter notebook converted to Python script.

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Advanced RAG Techniques - Vertex RAG Engine Retrieval Quality Evaluation and Hyperparameters Tuning

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_evaluation.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Frag-engine%2Frag_engine_evaluation.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/rag-engine/rag_engine_evaluation.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_evaluation.ipynb">
      <img width="32px" src="https://www.svgrepo.com/download/217753/github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

<div style="clear: both;"></div>

<b>Share to:</b>

<a href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_evaluation.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" alt="LinkedIn logo">
</a>

<a href="https://bsky.app/intent/compose?text=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_evaluation.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/7/7a/Bluesky_Logo.svg" alt="Bluesky logo">
</a>

<a href="https://twitter.com/intent/tweet?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_evaluation.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/5a/X_icon_2.svg" alt="X logo">
</a>

<a href="https://reddit.com/submit?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_evaluation.ipynb" target="_blank">
  <img width="20px" src="https://redditinc.com/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" alt="Reddit logo">
</a>

<a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_evaluation.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook logo">
</a>            
"""

"""
|           |                                         |
|-----------|---------------------------------------- |
| Author(s) | [Ed Tsoi](https://github.com/edtsoi430) |
"""

"""
## Overview

Retrieval Quality is arguably the most important component of a Retrieval Augmented Generation (RAG) application. Not only does it directly impact the quality of the generated response, in some cases poor retrieval could also lead to irrelevant, incomplete or hallucinated output.

This notebook aims to provide guidelines on:
> **You'll learn how to:**
> * Evaluate retrieval quality using the [BEIR-fiqa 2018 dataset](https://arxiv.org/abs/2104.08663) (or your own!).
> * Understand the impact of key parameters on retrieval performance. (e.g. embedding model, chunk size)
> * Tune hyperparameters to improve accuracy of the RAG system.

**Note:** This notebook assumes that you already have an understanding on how to implement a RAG system with Vertex AI RAG Engine. For more general instructions on how to use Vertex AI RAG Engine, please refer to the [RAG Engine API Documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/rag-api).

We'll explore how these hyperparameters influence retrieval:

| Parameter                 | Description                                                                         |
|---------------------------|-------------------------------------------------------------------------------------|
| Chunk Size                | Size of each chunk (in tokens). Affects granularity of retrieval.                   |
| Chunk Overlap             | Overlap between chunks. Helps capture relevant information across chunk boundaries. |
| Top K                     | Maximum number of retrieved contexts.  Balance recall and precision.                |
| Vector Distance threshold | Filters contexts based on similarity.  A stricter threshold prioritizes precision.  |
| Embedding model           | Model used to convert text to embeddings. Significantly impacts retrieval accuracy. |

### How exactly could we use this notebook to improve the RAG system?

* **Hyperparameters Tuning:** There are a couple of hyperparameters that could impact retrieval quality:

| Parameter | Description |
|------------|----------------------|
| Chunk Size | When documents are ingested into an index, they are split into chunks. The `chunk_size` parameter (in tokens) specifies the size of each chunk. |
| Chunk Overlap |  By default, documents are split into chunks with a certain amount of overlap to improve relevance and retrieval quality. |
| Top K | Controls the maximum number of contexts that are retrieved. |
| Vector Distance threshold | Only contexts with a distance smaller than the threshold are considered. |
| Embedding model | The embedding model used to convert input text into embeddings for retrieval.|

You may use this notebook to evaluate your retrieval quality, and see how changing certain parameters (top k, chunk size) impact or improve your retrieval quality (`recall@k`, `precision@k`, `ndcg@k`).

* **Response Quality Evaluation:** Once you have optimized the retrieval metrics, you can understand how it impacts response quality using the [Evaluation Service API Notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluate_rag_gen_ai_evaluation_service_sdk.ipynb)

"""

"""
## Get started
"""

"""
### Install Vertex AI SDK and other required packages

"""

%pip install --upgrade --user --quiet google-cloud-aiplatform beir

"""
### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.
"""

import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
# Output:
#   {'status': 'ok', 'restart': True}

"""
<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>

"""

"""
### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.
"""

import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()

"""
### Set Google Cloud project information and initialize the Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).
"""

# Use the environment variable if the user doesn't provide Project ID.
import os

import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)

!gcloud auth application-default login
!gcloud auth application-default set-quota-project {PROJECT_ID}
!gcloud config set project {PROJECT_ID}

"""
### Import libraries
"""

from collections.abc import MutableSequence
import math
import os
import re
import time

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from google.cloud import storage
from google.cloud.aiplatform_v1beta1.types import Context, RetrieveContextsResponse
import numpy as np
from tqdm import tqdm
from vertexai.preview import rag

"""
### Define helper function for processing dataset.
"""

def convert_beir_to_rag_corpus(
    corpus: dict[str, dict[str, str]], output_dir: str
) -> None:
    """
    Convert a BEIR corpus to Vertex RAG corpus format with a maximum of 10,000
    files per subdirectory.

    For each document in the BEIR corpus, we will create a new txt where:
      * doc_id will be the file name
      * doc_content will be the document text prepended by title (if any).

    Args:
      corpus: BEIR corpus
      output_dir (str): Directory where the converted corpus will be saved

    Returns:
      None (will write output to disk)
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    file_count, subdir_count = 0, 0
    current_subdir = os.path.join(output_dir, f"{subdir_count}")
    os.makedirs(current_subdir, exist_ok=True)

    # Convert each file in the corpus
    for doc_id, doc_content in corpus.items():
        # Combine title and text (if title exists)
        full_text = doc_content.get("title", "")
        if full_text:
            full_text += "\n\n"
        full_text += doc_content["text"]

        # Create a new file for each file.
        file_path = os.path.join(current_subdir, f"{doc_id}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(full_text)

        file_count += 1

        # Create a new subdirectory if the current one has reached the limit
        if file_count >= 10000:
            subdir_count += 1
            current_subdir = os.path.join(output_dir, f"{subdir_count}")
            os.makedirs(current_subdir, exist_ok=True)
            file_count = 0

    print(f"Conversion complete. {len(corpus)} files saved in {output_dir}")


def count_files_in_gcs_bucket(gcs_path: str) -> int:
    """
    Counts the number of files in a Google Cloud Storage path,
    excluding directories and hidden files.

    Args:
      gcs_path: The full GCS path, including the bucket name and any prefix.
       * Example: 'gs://my-bucket/my-folder'

    Returns:
      The number of files in the GCS path.
    """

    # Split the GCS path into bucket name and prefix
    bucket_name, prefix = gcs_path.replace("gs://", "").split("/", 1)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    count = 0
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if not blob.name.endswith("/") and not any(
            part.startswith(".") for part in blob.name.split("/")
        ):  # Exclude directories and hidden files
            count += 1

    return count


def count_directories_after_split(gcs_path: str) -> int:
    """
    Counts the number of directories in a Google Cloud Storage path.

    Args:
      gcs_path: The full GCS path, including the bucket name and any prefix.

    Returns:
      The number of directories in the GCS path.
    """
    num_files_in_path = count_files_in_gcs_bucket(gcs_path)
    num_directories = math.ceil(num_files_in_path / 10000)
    return num_directories


def import_rag_files_from_gcs(
    paths: list[str], chunk_size: int, chunk_overlap: int, corpus_name: str
) -> None:
    """Imports files from Google Cloud Storage to a RAG corpus.

    Args:
      paths: A list of GCS paths to import files from.
      chunk_size: The size of each chunk to import.
      chunk_overlap: The overlap between consecutive chunks.
      corpus_name: The name of the RAG corpus to import files into.

    Returns:
      None
    """
    total_imported, total_num_of_files = 0, 0

    for path in paths:
        num_files_to_be_imported = count_files_in_gcs_bucket(path)
        total_num_of_files += num_files_to_be_imported
        max_retries, attempt, imported = 10, 0, 0
        while attempt < max_retries and imported < num_files_to_be_imported:
            response = rag.import_files(
                corpus_name,
                [path],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                timeout=20000,
                max_embedding_requests_per_min=1400,
            )
            imported += response.imported_rag_files_count or 0
            attempt += 1
        total_imported += imported

    print(f"{total_imported} files out of {total_num_of_files} imported!")

"""
# For step 1, please choose only one of the following options:
- **1.1 (Option A, Recommended):** Create RagCorpus and perform data ingestion using the provided public GCS bucket (BEIR-fiqa dataset only).

- **1.2 (Option B):** Create RAG Corpus, choose a custom beir dataset and upload/ingest data into the RagCorpus on your own.

- **1.3 (Option C):** Bring your own existing `RagCorpus` (insert `RAG_CORPUS_ID` here).

**Do not run all these cells together.**
"""

"""
# 1.1 - Option A (Recommended): Create RagCorpus and perform data ingestion using the provided public GCS bucket (BEIR-fiqa dataset only).
* This option is recommended to save you time from having to upload evaluation dataset to GCS before we import them into the `RagCorpus`.
* However, if you would like more flexibility on which BEIR dataset to use, you could go with option B below to upload data to your desired GCS location.
* If you would like to bring your own rag corpus, simply skip to Option C and specify the rag corpus id.
"""

"""
### Create a `RagCorpus` with the specified configuration (for evaluation)
"""

# See the list of current supported embedding models here: https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview#supported-embedding-models
# Select embedding model as desired.
embedding_model_config = rag.EmbeddingModelConfig(
    publisher_model="publishers/google/models/text-embedding-005"  # @param {type:"string", isTemplate: true},
)

rag_corpus = rag.create_corpus(
    display_name="test-corpus",
    description="A test corpus where we import the BEIR-FiQA-2018 dataset",
    embedding_model_config=embedding_model_config,
)

print(rag_corpus)

"""
### Copy beir-fiqa dataset from the public path to a storage bucket in your project.
"""

CURRENT_BUCKET_PATH = "gs://<INSERT_GCS_PATH_HERE>"  # @param {type:"string"},

PUBLIC_BEIR_FIQA_GCS_PATH = (
    "gs://github-repo/generative-ai/gemini/rag-engine/rag_engine_evaluation/beir-fiqa"
)

!gsutil -m rsync -r -d $PUBLIC_BEIR_FIQA_GCS_PATH $CURRENT_BUCKET_PATH

"""
### Import evaluation dataset files into `RagCorpus` (configure chunk size, chunk overlap etc as desired)
"""

num_subdirectories = count_directories_after_split(CURRENT_BUCKET_PATH)
paths = [CURRENT_BUCKET_PATH + f"/{i}/" for i in range(num_subdirectories)]

chunk_size = 512  # @param {type:"integer"}
chunk_overlap = 102  # @param {type:"integer"}

import_rag_files_from_gcs(
    paths=paths,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    corpus_name=rag_corpus.name,
)
# Output:
#   57638 files out of 57638 imported!


"""
# 1.2 - Option B: Create RAG Corpus, choose a custom beir dataset and upload/ingest data into the RagCorpus on your own.

* Choose this option if you would like to have more flexibility on which dataset to use. The public, uploaded data in option 1.1 is for `BEIR-fiqa` only.
* If you would like to bring your own existing `RagCorpus` (with imported files), skip to Option C below.
"""

"""
### Create a `RagCorpus` with the specified configuration.
"""

# See the list of current supported embedding models here: https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview#supported-embedding-models
# You may adjust the embedding model here if you would like.
embedding_model_config = rag.EmbeddingModelConfig(
    publisher_model="publishers/google/models/text-embedding-005"  # @param {type:"string", isTemplate: true},
)

rag_corpus = rag.create_corpus(
    display_name="test-corpus",
    description="A test corpus where we import the BEIR-FiQA-2018 dataset",
    embedding_model_config=embedding_model_config,
)

print(rag_corpus)

"""
### Load BEIR Fiqa dataset (test split).
- Configure dataset of choice.
"""

# Download and load a BEIR dataset
dataset = "fiqa"  # @param ["arguana", "climate-fever", "cqadupstack", "dbpedia-entity", "fever", "fiqa", "germanquad", "hotpotqa", "mmarco", "mrtydi", "msmarco-v2", "msmarco", "nfcorpus", "nq-train", "nq", "quora", "scidocs", "scifact", "trec-covid-beir", "trec-covid-v2", "trec-covid", "vihealthqa", "webis-touche2020"]
url = (
    f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
)
out_dir = "datasets"
data_path = util.download_and_unzip(url, out_dir)

# Load the dataset
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
print(
    f"Successfully loaded the {dataset} dataset with {len(corpus)} files and {len(queries)} queries!"
)
# Output:
#   datasets/fiqa.zip:   0%|          | 0.00/17.1M [00:00<?, ?iB/s]
#     0%|          | 0/57638 [00:00<?, ?it/s]
#   Successfully loaded the fiqa dataset with 57638 files and 648 queries!


"""
### Convert BEIR corpus to `RagCorpus` format and upload to GCS bucket.
"""

CONVERTED_DATASET_PATH = f"/converted_dataset_{dataset}"
# Convert BEIR corpus to RAG format.
convert_beir_to_rag_corpus(corpus, CONVERTED_DATASET_PATH)

"""
#### Create a test bucket for uploading BEIR evaluation dataset to (or use an existing bucket of your choice).
"""

# Optionally rename bucket name here.
BUCKET_NAME = "beir-test-bucket"  # @param {type: "string"}
!gsutil mb gs://{BUCKET_NAME}

"""
#### Upload to specified GCS bucket (Modify the GCS bucket path to desired location)
"""

GCS_BUCKET_PATH = "gs://{BUCKET_NAME}/beir-fiqa"  # @param {type: "string"}

!echo "Uploading files from ${CONVERTED_DATASET_PATH} to ${GCS_BUCKET_PATH}"
# Upload RAG format dataset to GCS bucket.
!gsutil -m rsync -r -d $CONVERTED_DATASET_PATH $GCS_BUCKET_PATH

"""
### Import evaluation dataset files into `RagCorpus`.
"""

num_subdirectories = count_directories_after_split(GCS_BUCKET_PATH)
paths = [GCS_BUCKET_PATH + f"/{i}/" for i in range(num_subdirectories)]

chunk_size = 512  # @param {type:"integer"}
chunk_overlap = 102  # @param {type:"integer"}

import_rag_files_from_gcs(
    paths=paths,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    corpus_name=rag_corpus.name,
)

"""
# 1.3 - Option C: Bring your own existing `RagCorpus` (insert `RAG_CORPUS_ID` here).
"""

# Specify your rag corpus ID here that you want to use.
RAG_CORPUS_ID = ""  # @param {type: "string"}

rag_corpus = rag.get_corpus(
    name=f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{RAG_CORPUS_ID}"
)

print(rag_corpus)

"""
# 2. Run Retrieval Quality Evaluation

For Retrieval Quality Evaluation, we focus on the following metrics:

- **Recall@k:**
  - Measures how many of the relevant documents/chunks are successfully retrieved within the top k results
  - Helps evaluate the retrieval component's ability to find ALL relevant information
- **Precision@k:**
  - Measures the proportion of retrieved documents that are actually relevant within the top k results
  - Helps evaluate how "focused" your retrieval is
- **nDCG@K:**
  - Measures both relevance AND ranking quality
  - Takes into account the position of relevant documents

Follow the Notebook to get these metrics numbers for your configurations, and to optimize your settings.
"""

"""
### Define evaluation helper function.
"""

def extract_doc_id(file_path: str) -> str | None:
    """Extracts the document ID (filename without extension) from a file path.

    Handles various potential file name formats and extensions
    like .txt, .pdf, .html, etc.

    Args:
      file_path: The path to the file.

    Returns:
      The document ID (filename without extension) extracted from the file path.
    """
    try:
        # Split the path by directory separators
        parts = file_path.split("/")
        # Get the filename
        filename = parts[-1]
        # Remove the extension (if any)
        filename = re.sub(r"\.\w+$", "", filename)  # Removes .txt, .pdf, .html, etc.
        return filename
    except:
        pass  # Handle any unexpected errors during extraction
    return None


# RAG Engine helper function to extract doc_id, snippet, and score.


def extract_retrieval_details(
    response: RetrieveContextsResponse,
) -> tuple[str, str, float]:
    """Extracts the document ID, snippet, and score from a retrieval response.

    Args:
      response: The retrieval response object.

    Returns:
      A tuple containing the document ID, retrieved snippet, and distance score.
    """
    doc_id = extract_doc_id(response.source_uri)
    retrieved_snippet = response.text
    distance = response.distance
    return (doc_id, retrieved_snippet, distance)


# RAG Engine helper function for retrieval.


def rag_api_retrieve(
    query: str, corpus_name: str, top_k: int
) -> MutableSequence[Context]:
    """Retrieves relevant contexts from a RAG corpus using the RAG API.

    Args:
      query: The query text.
      corpus_name: The name of the RAG corpus, in the format of "projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{CORPUS_ID}".
      top_k: The number of top results to retrieve.

    Returns:
      A list of retrieved contexts.
    """
    return rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
        text=query,
        similarity_top_k=top_k,
        vector_distance_threshold=0.5,
    ).contexts.contexts


def calculate_document_level_recall_precision(
    retrieved_response: MutableSequence[Context], cur_qrel: dict[str, int]
) -> tuple[float, float]:
    """Calculates the recall and precision for a list of retrieved contexts.

    Args:
      retrieved_response: A list of retrieved contexts.
      cur_qrel: A dictionary of ground truth relevant documents for the current query.

    Returns:
      A tuple containing the recall and precision scores.
    """
    if not retrieved_response:
        return (0, 0)

    relevant_retrieved_unique = set()
    num_relevant_retrieved_snippet = 0
    for res in retrieved_response:
        doc_id, text, score = extract_retrieval_details(res)
        if doc_id in cur_qrel:
            relevant_retrieved_unique.add(doc_id)
            num_relevant_retrieved_snippet += 1
    recall = (
        len(relevant_retrieved_unique) / len(cur_qrel.keys())
        if len(cur_qrel.keys()) > 0
        else 0
    )
    precision = (
        num_relevant_retrieved_snippet / len(retrieved_response)
        if len(retrieved_response) > 0
        else 0
    )
    return (recall, precision)


def calculate_document_level_metrics(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k_values: list[int],
    corpus_name: str,
) -> None:
    """Calculates and prints the average recall, precision, and NDCG for a set of queries at different top_k values.

    Args:
      queries: A dictionary of queries with query IDs as keys and query text as values.
      qrels: A dictionary of ground truth relevant documents for each query.
      k_values: A list of top_k values to evaluate.
      corpus_name: The name of the RAG corpus, in the format of "projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{CORPUS_ID}".

    Returns:
      None
    """

    for top_k in k_values:
        start_time = time.time()
        total_recall, total_precision, total_ndcg = 0, 0, 0
        print(f"Computing metrics for top_k value: {top_k}")
        print(f"Total number of queries: {len(queries)}")
        for query_id, query in tqdm(
            queries.items(),
            total=len(queries),
            desc=f"Processing Queries (top_k={top_k})",
        ):
            response = rag_api_retrieve(query, corpus_name, top_k)

            recall, precision = calculate_document_level_recall_precision(
                response, qrels[query_id]
            )
            ndcg = ndcg_at_k(response, qrels[query_id], top_k)

            total_recall += recall
            total_precision += precision
            total_ndcg += ndcg

        end_time = time.time()
        execution_time = end_time - start_time
        num_queries = len(queries)
        average_recall, average_precision, average_ndcg = (
            total_recall / num_queries,
            total_precision / num_queries,
            total_ndcg / num_queries,
        )
        print(f"\nAverage Recall@{top_k}: {average_recall:.4f}")
        print(f"Average Precision@{top_k}: {average_precision:.4f}")
        print(f"Average nDCG@{top_k}: {average_ndcg:.4f}")
        print(f"Execution time: {execution_time} seconds.")
        print("=============================================")


def dcg_at_k_with_zero_padding_if_needed(r: list[int], k: int) -> float:
    """Calculates the Discounted Cumulative Gain (DCG) at a given rank k.

    Args:
      r: A list of relevance scores.
      k: The rank at which to calculate DCG.

    Returns:
      The DCG at rank k.
    """
    r = np.asarray(r)[:k]
    if r.size:
        # Pad with zeros if r is shorter than k
        if r.size < k:
            r = np.pad(r, (0, k - r.size))
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, k + 2)))
    return 0.0


def ndcg_at_k(
    retriever_results: MutableSequence[Context],
    ground_truth_relevances: dict[str, int],
    k: int,
) -> float:
    """Calculates the Normalized Discounted Cumulative Gain (NDCG) at a given rank k.

    Args:
      retriever_results: A list of retrieved results.
      ground_truth_relevances: A dictionary of ground truth relevance scores for each document.
      k: The rank at which to calculate NDCG.

    Returns:
      The NDCG at rank k.
    """
    if not retriever_results:
        return 0

    # Prepare retriever results
    retrieved_relevances = []
    for res in retriever_results[:k]:
        doc_id, text, score = extract_retrieval_details(res)
        if doc_id in ground_truth_relevances:
            retrieved_relevances.append(ground_truth_relevances[doc_id])
        else:
            retrieved_relevances.append(0)  # Assume irrelevant if not in ground truth

    # Calculate DCG
    dcg = dcg_at_k_with_zero_padding_if_needed(retrieved_relevances, k)
    # Calculate IDCG
    ideal_relevances = sorted(ground_truth_relevances.values(), reverse=True)
    idcg = dcg_at_k_with_zero_padding_if_needed(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0

"""
### Run Retrieval Quality Evaluation.
"""

calculate_document_level_metrics(
    queries, qrels, [5, 10, 100], corpus_name=rag_corpus.name
)
# Output:
#   Computing metrics for top_k value: 5

#   Total number of queries: 648

#   Processing Queries (top_k=5): 100%|██████████| 648/648 [44:47<00:00,  4.15s/it]

#   

#   Average Recall@5: 0.5608

#   Average Precision@5: 0.2713

#   Average nDCG@5: 0.4450

#   Execution time: 2687.608230829239 seconds.

#   =============================================

#   Computing metrics for top_k value: 10

#   Total number of queries: 648

#   Processing Queries (top_k=10): 100%|██████████| 648/648 [37:31<00:00,  3.48s/it]

#   

#   Average Recall@10: 0.6571

#   Average Precision@10: 0.1679

#   Average nDCG@10: 0.4039

#   Execution time: 2251.886693954468 seconds.

#   =============================================

#   Computing metrics for top_k value: 100

#   Total number of queries: 648

#   Processing Queries (top_k=100): 100%|██████████| 648/648 [38:48<00:00,  3.59s/it]
#   

#   Average Recall@100: 0.8801

#   Average Precision@100: 0.0253

#   Average nDCG@100: 0.2592

#   Execution time: 2328.4095141887665 seconds.

#   =============================================

#   


"""
# 3. Next steps
* Once we're done with evaluation, we should carefully examine the metrics number are tune the hypeparameters. Below are some suggestions on how to optimize the hyperparameters to get the best retrieval quality.

### How to optimize Recall:
* If your recall metrics number is too low, consider the following steps:
  * **Reducing chunk size:** Sometimes important information might be buried within large chunks, making it more difficult to retrieve relevant context. Try reducing the chunk size.
  * **Increasing chunk overlap:** If the chunk overlap is too small, some relevant information at the edge might be lost. Consider increasing the chunk overlap (chunk overlap of 20% of chunk size is generally a good start.)
  * **Increasing top-K:** If your top k is too small, the retriever might miss some relevant information due to a too restrictive context.

### How to optimize Precision:
* If your precision number is low, consider:
  * **Reducing top-K:** Your top k might be too large, adding a lot of unwanted noise to the retrieved contexts.
  * **Reducing chunk overlap:** Sometimes, too large of a chunk overlap could result in duplicate information.
  * **Increasing chunk size:** If your chunk size is too small, it might lack sufficient context resulting in a low precision score.

### How to optimize nDCG:
* If your nDCG number is low, consider:
  * **Changing your embedding model:** your embedding model might not capturing relevance well. Consider using a different embedding model (e.g. if your documents are multilingual, consider using a mulilingual embedding model). For more information on the currently supported embedding models, see documentation [here](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview#supported-embedding-models).

### Evaluate Response Quality
* If you want to evaluate response quality (generated answers) on top of retrieval quality, please refer to the [Gen AI Evaluation Service - RAG Evaluation Notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluate_rag_gen_ai_evaluation_service_sdk.ipynb)

"""

"""
# 4. Cleaning up (Delete `RagCorpus`)

Once we are done with evaluation, we should clean up the `RagCorpus` to free up resources since we don't need it anymore.
"""

rag.delete_corpus(rag_corpus.name)



================================================
FILE: gemini/rag-engine/rag_engine_feature_store.ipynb
================================================
# Jupyter notebook converted to Python script.

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Vertex AI RAG Engine with Vertex AI Feature Store

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_feature_store.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Frag-engine%2Frag_engine_feature_store.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/rag-engine/rag_engine_feature_store.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/bigquery/import?url=https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_feature_store.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/bigquery/v1/32px.svg" alt="BigQuery Studio logo"><br> Open in BigQuery Studio
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_feature_store.ipynb">
      <img width="32px" src="https://www.svgrepo.com/download/217753/github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

<div style="clear: both;"></div>

<b>Share to:</b>

<a href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_feature_store.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" alt="LinkedIn logo">
</a>

<a href="https://bsky.app/intent/compose?text=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_feature_store.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/7/7a/Bluesky_Logo.svg" alt="Bluesky logo">
</a>

<a href="https://twitter.com/intent/tweet?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_feature_store.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/5a/X_icon_2.svg" alt="X logo">
</a>

<a href="https://reddit.com/submit?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_feature_store.ipynb" target="_blank">
  <img width="20px" src="https://redditinc.com/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" alt="Reddit logo">
</a>

<a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_feature_store.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook logo">
</a>            
"""

"""
| | |
|-|-|
| Author(s) | [Holt Skinner](https://github.com/holtskinner) |
"""

"""
## Overview

This notebook illustrates how to use [Vertex AI RAG Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview) with [Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore/latest/overview) as a vector database.

RAG Engine uses a built-in vector database powered by Spanner to store and manage vector representations of text documents. The vector database retrieves relevant documents based on the documents' semantic similarity to a given query.

By integrating Vertex AI Feature Store as an additional vector database, RAG Engine can use Vertex AI Feature Store to handle large data volumes with low latency, which helps to improve the performance and scalability of your RAG applications.

For more information, refer to the [official documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/use-vertexai-vector-search).

For more details on RAG corpus/file management and detailed support please visit [Vertex AI RAG Engine API](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/rag-api)
"""

"""
## Get started
"""

"""
### Install Vertex AI SDK and other required packages

"""

%pip install --upgrade --user --quiet google-cloud-aiplatform google-cloud-bigquery

"""
### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.
"""

import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)

"""
<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>

"""

"""
### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.
"""

import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()

"""
### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).
"""

# Use the environment variable if the user doesn't provide Project ID.
import os

import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type:"string", isTemplate: true}
if PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)

"""
### Import Libraries
"""

from google.cloud import bigquery
from vertexai.preview import rag
from vertexai.preview.generative_models import GenerativeModel, Tool
from vertexai.resources.preview import feature_store

"""
## Set up Vertex AI Feature Store

Vertex AI Feature Store, a managed cloud-native service, is an essential component of Vertex AI. It simplifies machine learning (ML) feature management and online serving by letting you manage feature data within a BigQuery table or view. This enables low-latency online feature serving.

For `FeatureOnlineStore` instances created with optimized online serving, you
can take advantage of a vector similarity search to retrieve a list of
semantically similar or related entities, which are known as
*approximate nearest neighbors*.

The following sections show you how to set up a Vertex AI Feature Store instance for your RAG application.

"""

"""
### Create a BigQuery table schema

Use the Cloud Console or the code below to create a BigQuery table schema. It
must contain the following fields to serve as the data source.

| Field name | Data type | Status |
|-------------|-----------|--------|
| `corpus_id` | `String` | Required |
| `file_id` | `String` | Required |
| `chunk_id` | `String` | Required |
| `chunk_data_type` |`String` | Nullable |
| `chunk_data` | `String` | Nullable |
| `file_original_uri` | `String` | Nullable |
| `embeddings` | `Float` | Repeated |

"""

client = bigquery.Client(project=PROJECT_ID)

# Define dataset and table name
dataset_id = "input_us_central1"  # @param {type:"string"}
table_id = "rag_source_new"  # @param {type:"string"}

schema = [
    bigquery.SchemaField("corpus_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("file_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("chunk_data_type", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("chunk_data", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("file_original_uri", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("embeddings", "FLOAT64", mode="REPEATED"),
]

dataset_ref = bigquery.DatasetReference(PROJECT_ID, dataset_id)

try:
    dataset = client.get_dataset(dataset_ref)
    print(f"Dataset {dataset_id} already exists.")
except Exception:
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"  # Set the location (optional, adjust if needed)
    dataset = client.create_dataset(dataset)
    print(f"Created dataset {dataset.dataset_id}")

table_ref = dataset_ref.table(table_id)
table = client.create_table(bigquery.Table(table_ref, schema=schema))
print(f"Created table {PROJECT_ID}.{dataset_id}.{table_id}")

BIGQUERY_TABLE = f'bq://{table.full_table_id.replace(":", ".")}'

"""
### Provision a `FeatureOnlineStore` instance

To enable online serving of features, use the `CreateFeatureOnlineStore` API to set up a `FeatureOnlineStore` instance. If you
are provisioning a `FeatureOnlineStore` for the first time, the operation might
take approximately five minutes to complete.
"""

FEATURE_ONLINE_STORE_ID = "your_feature_online_store_id"  # @param {type: "string"}

fos = feature_store.FeatureOnlineStore.create_optimized_store(FEATURE_ONLINE_STORE_ID)

"""
### Create a `FeatureView` resource

To connect the BigQuery table, which stores the feature data source, to
the `FeatureOnlineStore` instance, call the `CreateFeatureView` API to create a
`FeatureView` resource. When you create a `FeatureView` resource, choose the
default distance metric `DOT_PRODUCT_DISTANCE`, which is defined as the
negative of the dot product (smaller `DOT_PRODUCT_DISTANCE` indicates higher
similarity).
"""

FEATURE_VIEW_ID = "your_feature_view_id"  # @param {type: "string"}
fv = fos.create_feature_view(
    name=FEATURE_VIEW_ID,
    source=feature_store.utils.FeatureViewVertexRagSource(uri=BIGQUERY_TABLE),
)

# Check that Feature View was created
print(fv)

"""
## Use Vertex AI Feature Store in RAG Engine

After the Feature Store instance is set up, the following
sections show you how to set it up as the vector database to use with the RAG
application.
"""

"""
### Set the vector database to create a RAG corpus

To create the RAG corpus, you must use `FEATURE_VIEW_RESOURCE_NAME`. The
RAG corpus is created and automatically associated with the
Vertex AI Feature Store instance.

RAG APIs use the generated `rag_corpus_id` to handle the data upload to the Vertex AI Feature Store
instance and to retrieve relevant contexts from the `rag_corpus_id`.
"""

vector_db = rag.VertexFeatureStore(resource_name=fv.resource_name)

# Name your corpus
DISPLAY_NAME = "Feature Store Corpus"  # @param  {type:"string"}

# Create RAG Corpus
rag_corpus = rag.create_corpus(display_name=DISPLAY_NAME, vector_db=vector_db)
print(f"Created RAG Corpus resource: {rag_corpus.name}")

rag_corpus

"""
## Import files into the BigQuery table using the RAG API

Use the `ImportRagFiles` API to import files from Google Cloud Storage or
Google Drive into the BigQuery table of the Vertex AI Feature Store
instance. The files are embedded and stored in the BigQuery table.

Remember to grant "Viewer" access to the "Vertex RAG Data Service Agent" (with the format of `service-{project_number}@gcp-sa-vertex-rag.iam.gserviceaccount.com`) for your Google Cloud Storage bucket
"""

GCS_BUCKET = "cloud-samples-data/gen-app-builder/search/cymbal-bank-employee"  # @param {type:"string"}

response = rag.import_files(  # noqa: F704
    corpus_name=rag_corpus.name,
    paths=[GCS_BUCKET],
    chunk_size=512,
    chunk_overlap=50,
)

# Check the files just imported. It may take a few seconds to process the imported files.
rag.list_files(corpus_name=rag_corpus.name)

"""
### Run a synchronization process to construct a `FeatureOnlineStore` index {:#run-sync-process}

After uploading your data into the BigQuery table, run a
synchronization process to make your data available for online serving. You must
generate a `FeatureOnlineStore` index using the `FeatureView`, and the
synchronization process might take 20 minutes to complete.

"""

feature_view_sync = fv.sync()
feature_view_sync

# Optional: Wait for sync to complete
feature_view_sync.wait()

"""
## Use your RAG Corpus to add context to your Gemini queries

When retrieved contexts similarity distance < `vector_distance_threshold`, the contexts (from `RagStore`) will be used for content generation.
"""

rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=rag_corpus.name,  # Currently only 1 corpus is allowed.
                )
            ],
            similarity_top_k=10,
            vector_distance_threshold=0.4,
        ),
    )
)

rag_model = GenerativeModel("gemini-2.0-flash", tools=[rag_retrieval_tool])

GENERATE_CONTENT_PROMPT = "What is RAG and why it is helpful?"  # @param {type:"string"}

response = rag_model.generate_content(GENERATE_CONTENT_PROMPT)

response.text

"""
## Using other generation APIs with RAG Retrieval Tool

The retrieved contexts can be passed to any SDK or model generation API to generate final results.
"""

RETRIEVAL_QUERY = "What is RAG and why it is helpful?"  # @param {type:"string"}

response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(
            rag_corpus=rag_corpus.name,  # Currently only 1 corpus is allowed.
        )
    ],
    text=RETRIEVAL_QUERY,
    similarity_top_k=10,
)

# The retrieved context can be passed to any SDK or model generation API to generate final results.
retrieved_context = " ".join(
    [context.text for context in response.contexts.contexts]
).replace("\n", "")

retrieved_context

"""
## Cleaning up

Clean up resources created in this notebook.
"""

delete_rag_corpus = False  # @param {type:"boolean"}

if delete_rag_corpus:
    rag.delete_corpus(name=rag_corpus.name)



================================================
FILE: gemini/rag-engine/rag_engine_pinecone.ipynb
================================================
# Jupyter notebook converted to Python script.

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Vertex AI RAG Engine with Pinecone

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_pinecone.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Frag-engine%2Frag_engine_pinecone.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/rag-engine/rag_engine_pinecone.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_pinecone.ipynb">
      <img width="32px" src="https://www.svgrepo.com/download/217753/github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

<div style="clear: both;"></div>

<b>Share to:</b>

<a href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_pinecone.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" alt="LinkedIn logo">
</a>

<a href="https://bsky.app/intent/compose?text=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_pinecone.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/7/7a/Bluesky_Logo.svg" alt="Bluesky logo">
</a>

<a href="https://twitter.com/intent/tweet?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_pinecone.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/5a/X_icon_2.svg" alt="X logo">
</a>

<a href="https://reddit.com/submit?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_pinecone.ipynb" target="_blank">
  <img width="20px" src="https://redditinc.com/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" alt="Reddit logo">
</a>

<a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_pinecone.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook logo">
</a>            
"""

"""
| | |
|-|-|
| Author(s) | [Darshan Mehta](https://github.com/darshanmehta17) |
"""

"""
## Overview

This notebook illustrates how to use [Vertex AI RAG Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview) with [Pinecone](https://www.pinecone.io/) as a vector database.

For more information, refer to the [official documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/use-pinecone).

For more details on RAG corpus/file management and detailed support please visit [Vertex AI RAG Engine API](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/rag-api)
"""

"""
## Get started
"""

"""
### Install Vertex AI SDK and other required packages

"""

%pip install --upgrade --quiet google-cloud-aiplatform google-cloud-secret-manager "pinecone[grpc]" google-genai

"""
### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.
"""

import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)

"""
<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>

"""

"""
### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.
"""

import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()

"""
### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).
"""

# Use the environment variable if the user doesn't provide Project ID.
import os

from google import genai
import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

"""
## (Optional) Setup Pinecone Instance

In this section, we have some helper methods to help you setup your Pinecone instance.

Follow the [Pinecone Quickstart](https://docs.pinecone.io/guides/get-started/quickstart) to get an API Key.

This section is not required if you already have a Pinecone instance ready to use.
"""

"""
### Initialize the Pinecone Python client
"""

from pinecone import PodSpec, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone

# Set API Key
# Copy this value from your pinecone.io console
PINECONE_API_KEY = ""  # @param {type:"string"}

pc = Pinecone(api_key=PINECONE_API_KEY)

"""
### Create a Pinecone instance

Use the below section to create Pinecone indexes with a spec of your choice (serverless or pod). Read more about Pinecone indexes [here](https://docs.pinecone.io/guides/indexes/create-an-index).
"""

"""
#### Serverless Index
"""

# Index Configs
INDEX_NAME = ""  # @param {type:"string"}

# Choose a distance metric
DISTANCE_METRIC = (
    "cosine"  # @param ["cosine", "euclidean", "dotproduct"] {allow-input: true}
)

# This number should match the dimension size of the embedding model you choose
# for your RAG Corpus.
EMBEDDING_DIMENSION_SIZE = 768  # @param {"type":"number","placeholder":"768"}

CLOUD_PROVIDER = "gcp"  # @param ["gcp", "aws", "azure"] {allow-input: true}

# Choose the right region for your cloud provider of choice
# Refer https://docs.pinecone.io/guides/indexes/understanding-indexes#cloud-regions
CLOUD_REGION = "us-central1"  # @param {type:"string"}


# Create the index
pc.create_index(
    name=INDEX_NAME,
    dimension=EMBEDDING_DIMENSION_SIZE,
    metric=DISTANCE_METRIC,
    spec=ServerlessSpec(cloud=CLOUD_PROVIDER, region=CLOUD_REGION),
    deletion_protection="disabled",
)

"""
#### Pod-based Index

"""

# Index Configs
INDEX_NAME = ""  # @param {type:"string"}

# Choose a distance metric
DISTANCE_METRIC = (
    "cosine"  # @param ["cosine", "euclidean", "dotproduct"] {allow-input: true}
)

# This number should match the dimension size of the embedding model you choose
# for your RAG Corpus.
EMBEDDING_DIMENSION_SIZE = 768  # @param {"type":"number","placeholder":"768"}

# Choose the right environment for your cloud provider of choice
# Refer https://docs.pinecone.io/guides/indexes/understanding-indexes#pod-environments
ENVIRONMENT = "us-central1-gcp"  # @param {type:"string"}

# Choose the pod type
# Refer to https://docs.pinecone.io/guides/indexes/understanding-indexes#pod-based-indexes
POD_TYPE = "p1.x1"  # @param {type:"string"}

# Explore all the parameters you can play with for creating a pod index by
# following this page:
# https://docs.pinecone.io/reference/api/2024-07/control-plane/create_index
pc.create_index(
    name=INDEX_NAME,
    dimension=EMBEDDING_DIMENSION_SIZE,
    metric=DISTANCE_METRIC,
    spec=PodSpec(
        environment=ENVIRONMENT,
        pod_type=POD_TYPE,
        pods=1,
        metadata_config={
            "indexed": ["file_id"]  # This field is required for pod-based indexes.
        },
    ),
    deletion_protection="disabled",
)

"""
## (Optional) Setup Secret Manager

This section helps you add your Pinecone API key to your Google Cloud Secret Manager. This section is not required is you already have a secret with the API key ready to use.
"""

# Google Cloud project ID and the Pinecone API key will be used from the above sections.
# Choose your secret ID
SECRET_ID = ""  # @param {type:"string"}

from google.cloud import secretmanager

secretmanager_client = secretmanager.SecretManagerServiceClient()

# Create the secret.
secret = secretmanager_client.create_secret(
    parent=secretmanager_client.common_project_path(PROJECT_ID),
    secret_id=SECRET_ID,
    secret=secretmanager.Secret(
        replication=secretmanager.Replication(
            automatic=secretmanager.Replication.Automatic()
        )
    ),
)

# Add API key to the secret payload.
secret_version = secretmanager_client.add_secret_version(
    parent=secret.name,
    payload=secretmanager.SecretPayload(data=PINECONE_API_KEY.encode("UTF-8")),
)

print(f"Created secret and added first version: {secret_version.name}")

"""
## Get Service Account Information
"""

project_numbers = !gcloud projects list --filter="PROJECT_ID={PROJECT_ID}" --format="value(PROJECT_NUMBER)"
PROJECT_NUMBER = project_numbers[0]

# Construct your RAG Engine service account name
# Do not update this string since this is the name assigned to your service account.
SERVICE_ACCOUNT = f"service-{PROJECT_NUMBER}@gcp-sa-vertex-rag.iam.gserviceaccount.com"

"""
## Create a RAG corpus
"""

from google.genai.types import (
    GenerateContentConfig,
    Retrieval,
    Tool,
    VertexRagStore,
    VertexRagStoreRagResource,
)
from vertexai import rag

"""
### First RAG Corpus Only

If this is the first RAG Corpus in your project, you might not be able to provide the RAG Engine service account access to your secret resource. So this section first creates a RAG Corpus with an empty Pinecone config. With this call, the service account for your project is provisioned.

Next, it assigns the service account permissions to read your secret. Finally, it updates your RAG Corpus with the Pinecone index name and the secret resource name.
"""

"""
#### Create a RAG Corpus without Pinecone information
"""

# Start with empty Pinecone config.
vector_db = rag.Pinecone()

# Name your corpus
DISPLAY_NAME = ""  # @param  {type:"string"}

# Create RAG Corpus
rag_corpus = rag.create_corpus(
    display_name=DISPLAY_NAME, backend_config=rag.RagVectorDbConfig(vector_db=vector_db)
)
print(f"Created RAG Corpus resource: {rag_corpus.name}")

"""
#### Grant your RAG Engine service account access to your API key secret
"""

!gcloud secrets add-iam-policy-binding {secret.name} \
  --member="serviceAccount:{SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

"""
#### Call the `UpdateRagCorpus` API to add the Pinecone index name and API key secret to your RAG Corpus
"""

# Name of your created Pinecone Index
PINECONE_INDEX_NAME = ""  # @param  {type:"string"}

# Construct your updated Pinecone config.
vector_db = rag.Pinecone(index_name=PINECONE_INDEX_NAME, api_key=secret_version.name)

updated_rag_corpora = rag.update_corpus(
    corpus_name=rag_corpus.name,
    backend_config=rag.RagVectorDbConfig(vector_db=vector_db),
)
print(f"Updated RAG Corpus: {rag_corpus.name}")

"""
### Second RAG Corpus Onwards

In this case, since your service account is already generated, you can directly grant it permissions to access your secret resource containing the Pinecone API key as covered by the steps in the Setup Secret Manager section.

"""

"""
#### Grant your RAG Engine service account access to your API key secret
"""

!gcloud secrets add-iam-policy-binding {secret.name} \
  --member="serviceAccount:{SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

"""
#### Create a RAG Corpus with Pinecone information
"""

# Name of your created Pinecone Index
PINECONE_INDEX_NAME = ""  # @param  {type:"string"}
# Construct your Pinecone config.
vector_db = rag.Pinecone(index_name=PINECONE_INDEX_NAME, api_key=secret_version.name)

# Name your corpus
DISPLAY_NAME = ""  # @param  {type:"string"}

# Create RAG Corpus
rag_corpus = rag.create_corpus(
    display_name=DISPLAY_NAME, backend_config=rag.RagVectorDbConfig(vector_db=vector_db)
)
print(f"Created RAG Corpus resource: {rag_corpus.name}")

"""
## Upload a file to the corpus
"""

%%writefile test.txt

Here's a demo for Pinecone RAG.

rag_file = rag.upload_file(
    corpus_name=rag_corpus.name,
    path="test.txt",
    display_name="test.txt",
    description="my test",
)
print(f"Uploaded file to resource: {rag_file.name}")

"""
## Import files from Google Cloud Storage

Remember to grant "Viewer" access to the "Vertex RAG Data Service Agent" (with the format of `service-{project_number}@gcp-sa-vertex-rag.iam.gserviceaccount.com`) for your Google Cloud Storage bucket
"""

GCS_BUCKET = ""  # @param {type:"string", "placeholder": "your-gs-bucket"}

response = rag.import_files(  # noqa: F704
    corpus_name=rag_corpus.name,
    paths=[GCS_BUCKET],
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=512,
            chunk_overlap=50,
        )
    ),
)

# Check the files just imported. It may take a few seconds to process the imported files.
rag.list_files(corpus_name=rag_corpus.name)

"""
## Import files from Google Drive

Eligible paths can be:

- `https://drive.google.com/drive/folders/{folder_id}`
- `https://drive.google.com/file/d/{file_id}`

Remember to grant "Viewer" access to the "Vertex RAG Data Service Agent" (with the format of `service-{project_number}@gcp-sa-vertex-rag.iam.gserviceaccount.com`) for your Drive folder/files.

"""

FILE_ID = ""  # @param {type:"string", "placeholder": "your-file-id"}
FILE_PATH = f"https://drive.google.com/file/d/{FILE_ID}"

rag.import_files(
    corpus_name=rag_corpus.name,
    paths=[FILE_PATH],
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=1024,
            chunk_overlap=100,
        )
    ),
)

# Check the files just imported. It may take a few seconds to process the imported files.
rag.list_files(corpus_name=rag_corpus.name)

"""
## Use your RAG Corpus to add context to your Gemini queries

When retrieved contexts similarity distance < `vector_distance_threshold`, the contexts (from `RagStore`) will be used for content generation.
"""

MODEL_ID = "gemini-2.0-flash-001"

rag_retrieval_tool = Tool(
    retrieval=Retrieval(
        vertex_rag_store=VertexRagStore(
            rag_resources=[
                VertexRagStoreRagResource(
                    rag_corpus=rag_corpus.name  # Currently only 1 corpus is allowed.
                )
            ],
            similarity_top_k=10,
            vector_distance_threshold=0.4,
        )
    )
)

GENERATE_CONTENT_PROMPT = "What is RAG and why it is helpful?"  # @param {type:"string"}

response = client.models.generate_content(
    model=MODEL_ID,
    contents=GENERATE_CONTENT_PROMPT,
    config=GenerateContentConfig(tools=[rag_retrieval_tool]),
)

display(Markdown(response.text))

"""
## Using other generation API with Rag Retrieval Tool

The retrieved contexts can be passed to any SDK or model generation API to generate final results.
"""

RETRIEVAL_QUERY = "What is RAG and why it is helpful?"  # @param {type:"string"}

rag_resource = rag.RagResource(
    rag_corpus=rag_corpus.name,
    # Need to manually get the ids from rag.list_files.
    # rag_file_ids=[],
)

response = rag.retrieval_query(
    rag_resources=[rag_resource],  # Currently only 1 corpus is allowed.
    text=RETRIEVAL_QUERY,
    rag_retrieval_config=rag.RagRetrievalConfig(
        top_k=10,  # Optional
        filter=rag.Filter(
            vector_distance_threshold=0.5,  # Optional
        ),
    ),
)

# The retrieved context can be passed to any SDK or model generation API to generate final results.
retrieved_context = " ".join(
    [context.text for context in response.contexts.contexts]
).replace("\n", "")

retrieved_context

"""
## Cleaning up

Clean up resources created in this notebook.
"""

delete_rag_corpus = False  # @param {type:"boolean"}

if delete_rag_corpus:
    rag.delete_corpus(name=rag_corpus.name)



================================================
FILE: gemini/rag-engine/rag_engine_vector_search.ipynb
================================================
# Jupyter notebook converted to Python script.

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Vertex AI RAG Engine with Vertex AI Vector Search

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_vector_search.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Frag-engine%2Frag_engine_vector_search.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/rag-engine/rag_engine_vector_search.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_vector_search.ipynb">
      <img width="32px" src="https://www.svgrepo.com/download/217753/github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

<div style="clear: both;"></div>

<b>Share to:</b>

<a href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_vector_search.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" alt="LinkedIn logo">
</a>

<a href="https://bsky.app/intent/compose?text=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_vector_search.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/7/7a/Bluesky_Logo.svg" alt="Bluesky logo">
</a>

<a href="https://twitter.com/intent/tweet?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_vector_search.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/5a/X_icon_2.svg" alt="X logo">
</a>

<a href="https://reddit.com/submit?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_vector_search.ipynb" target="_blank">
  <img width="20px" src="https://redditinc.com/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" alt="Reddit logo">
</a>

<a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_vector_search.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook logo">
</a>            
"""

"""
| | |
|-|-|
| Author(s) | [Holt Skinner](https://github.com/holtskinner) |
"""

"""
## Overview

This notebook illustrates how to use [Vertex AI RAG Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview) with [Vertex AI Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview) as a vector database.

For more information, refer to the [official documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/use-vertexai-vector-search).

For more details on RAG corpus/file management and detailed support please visit [Vertex AI RAG Engine API](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/rag-api)
"""

"""
## Get started
"""

"""
### Install Vertex AI SDK and other required packages

"""

%pip install --upgrade --quiet google-cloud-aiplatform google-genai
# Output:
#   [33mWARNING: Skipping /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/google_cloud_aiplatform-1.50.0.dist-info due to invalid metadata entry 'name'[0m[33m

#   [0m[33mWARNING: Skipping /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/google_cloud_storage-2.16.0.dist-info due to invalid metadata entry 'name'[0m[33m

#   [0m[33mDEPRECATION: Loading egg at /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/fsspec-2024.3.1-py3.11.egg is deprecated. pip 24.3 will enforce this behaviour change. A possible replacement is to use pip for package installation.. Discussion can be found at https://github.com/pypa/pip/issues/12330[0m[33m

#   [0m[33mDEPRECATION: Loading egg at /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/google_cloud_documentai_toolbox-0.12.2a0-py3.11.egg is deprecated. pip 24.3 will enforce this behaviour change. A possible replacement is to use pip for package installation.. Discussion can be found at https://github.com/pypa/pip/issues/12330[0m[33m

#   [0m[33mDEPRECATION: Loading egg at /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/google_cloud_documentai_toolbox-0.11.1a0-py3.11.egg is deprecated. pip 24.3 will enforce this behaviour change. A possible replacement is to use pip for package installation.. Discussion can be found at https://github.com/pypa/pip/issues/12330[0m[33m

#   [0m[33mWARNING: Skipping /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/google_cloud_aiplatform-1.50.0.dist-info due to invalid metadata entry 'name'[0m[33m

#   [0m[33mWARNING: Skipping /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/google_cloud_storage-2.16.0.dist-info due to invalid metadata entry 'name'[0m[33m

#   [0m[33mWARNING: Skipping /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/google_cloud_aiplatform-1.50.0.dist-info due to invalid metadata entry 'name'[0m[33m

#   [0m[33mWARNING: Skipping /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/google_cloud_aiplatform-1.50.0.dist-info due to invalid metadata entry 'name'[0m[33m

#   [0m

#   [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.0[0m[39;49m -> [0m[32;49m25.0.1[0m

#   [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip3.11 install --upgrade pip[0m

#   Note: you may need to restart the kernel to use updated packages.


"""
### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.
"""

import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)

"""
<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>

"""

"""
### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.
"""

import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()

"""
### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).
"""

# Use the environment variable if the user doesn't provide Project ID.
import os

from google import genai
from google.cloud import aiplatform

PROJECT_ID = "[your-project-id]"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

aiplatform.init(project=PROJECT_ID, location=LOCATION)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

"""
## (Optional) Setup Vertex AI Vector Search index and index endpoint

In this section, we have some helper methods to help you setup your Vector Search index.

This section is not required if you already have a Vector Search index ready to use.

The index has to meet the following criteria:

1. `IndexUpdateMethod` must be `STREAM_UPDATE`, see [Create stream index]({{docs_path}}vector-search/create-manage-index#create-stream-index).

2. Distance measure type must be explicitly set to one of the following:

   * `DOT_PRODUCT_DISTANCE`
   * `COSINE_DISTANCE`

3. Dimension of the vector must be consistent with the embedding model you plan
   to use in the RAG corpus. Other parameters can be tuned based on
   your choices, which determine whether the additional parameters can be
   tuned.
"""

# create the index
my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="your_display_name",
    description="your_description",
    dimensions=768,
    approximate_neighbors_count=10,
    leaf_node_embedding_count=500,
    leaf_nodes_to_search_percent=7,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    feature_norm_type="UNIT_L2_NORM",
    index_update_method="STREAM_UPDATE",
)

"""
RAG Engine supports [public endpoints](https://cloud.google.com/vertex-ai/docs/vector-search/deploy-index-public).
"""

# create IndexEndpoint
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="your_display_name", public_endpoint_enabled=True
)

"""
Deploy the index to the index endpoint.

If it's the first time that you're deploying an index to an index endpoint, it
takes approximately 30 minutes to automatically build and initiate the backend
before the index can be stored. After the first deployment, the index is ready
in seconds. To see the status of the index deployment, open the
[**Vector Search Console**](https://console.cloud.google.com/vertex-ai/matching-engine/index-endpoints),
select the **Index endpoints** tab, and choose your index endpoint.

Identify the resource name of your index and index endpoint, which have the
following the formats:

* `projects/${PROJECT_ID}/locations/${LOCATION_ID}/indexes/${INDEX_ID}`
* `projects/${PROJECT_ID}/locations/${LOCATION_ID}/indexEndpoints/${INDEX_ENDPOINT_ID}`.

If you aren't sure about the resource name, you can use the following command to
check:
"""

print(my_index_endpoint.resource_name)
print(my_index.resource_name)

# Deploy Index
my_index_endpoint.deploy_index(
    index=my_index, deployed_index_id="your_deployed_index_id"
)

"""
## Use Vertex AI Vector Search in RAG Engine

After the Vector Search instance is set up, follow the steps in this section to set the Vector Search instance as the vector database for the RAG application.

"""

"""
### Set the vector database to create a RAG corpus
"""

from google.genai.types import (
    GenerateContentConfig,
    Retrieval,
    Tool,
    VertexRagStore,
    VertexRagStoreRagResource,
)
from vertexai import rag

vector_db = rag.VertexVectorSearch(
    index=my_index.resource_name, index_endpoint=my_index_endpoint.resource_name
)

# Name your corpus
DISPLAY_NAME = ""  # @param  {type:"string"}

# Create RAG Corpus
rag_corpus = rag.create_corpus(
    display_name=DISPLAY_NAME, backend_config=rag.RagVectorDbConfig(vector_db=vector_db)
)
print(f"Created RAG Corpus resource: {rag_corpus.name}")

"""
## Upload a file to the corpus
"""

%%writefile test.txt

Here's a demo for Vertex AI Vector Search RAG.

rag_file = rag.upload_file(
    corpus_name=rag_corpus.name,
    path="test.txt",
    display_name="test.txt",
    description="my test",
)
print(f"Uploaded file to resource: {rag_file.name}")

"""
## Import files from Google Cloud Storage

Remember to grant "Viewer" access to the "Vertex RAG Data Service Agent" (with the format of `service-{project_number}@gcp-sa-vertex-rag.iam.gserviceaccount.com`) for your Google Cloud Storage bucket
"""

GCS_BUCKET = ""  # @param {type:"string", "placeholder": "your-gs-bucket"}

response = rag.import_files(  # noqa: F704
    corpus_name=rag_corpus.name,
    paths=[GCS_BUCKET],
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=512,
            chunk_overlap=50,
        )
    ),
)

# Check the files just imported. It may take a few seconds to process the imported files.
rag.list_files(corpus_name=rag_corpus.name)

"""
## Import files from Google Drive

Eligible paths can be:

- `https://drive.google.com/drive/folders/{folder_id}`
- `https://drive.google.com/file/d/{file_id}`

Remember to grant "Viewer" access to the "Vertex RAG Data Service Agent" (with the format of `service-{project_number}@gcp-sa-vertex-rag.iam.gserviceaccount.com`) for your Drive folder/files.

"""

FILE_ID = ""  # @param {type:"string", "placeholder": "your-file-id"}
FILE_PATH = f"https://drive.google.com/file/d/{FILE_ID}"

rag.import_files(
    corpus_name=rag_corpus.name,
    paths=[FILE_PATH],
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=1024,
            chunk_overlap=100,
        )
    ),
)

# Check the files just imported. It may take a few seconds to process the imported files.
rag.list_files(corpus_name=rag_corpus.name)

"""
## Use your RAG Corpus to add context to your Gemini queries

When retrieved contexts similarity distance < `vector_distance_threshold`, the contexts (from `RagStore`) will be used for content generation.
"""

MODEL_ID = "gemini-2.0-flash-001"

rag_retrieval_tool = Tool(
    retrieval=Retrieval(
        vertex_rag_store=VertexRagStore(
            rag_resources=[
                VertexRagStoreRagResource(
                    rag_corpus=rag_corpus.name  # Currently only 1 corpus is allowed.
                )
            ],
            similarity_top_k=10,
            vector_distance_threshold=0.4,
        )
    )
)

GENERATE_CONTENT_PROMPT = "What is RAG and why it is helpful?"  # @param {type:"string"}

response = client.models.generate_content(
    model=MODEL_ID,
    contents=GENERATE_CONTENT_PROMPT,
    config=GenerateContentConfig(tools=[rag_retrieval_tool]),
)

display(Markdown(response.text))

"""
## Using other generation API with Rag Retrieval Tool

The retrieved contexts can be passed to any SDK or model generation API to generate final results.
"""

RETRIEVAL_QUERY = "What is RAG and why it is helpful?"  # @param {type:"string"}

rag_resource = rag.RagResource(
    rag_corpus=rag_corpus.name,
    # Need to manually get the ids from rag.list_files.
    # rag_file_ids=[],
)

response = rag.retrieval_query(
    rag_resources=[rag_resource],  # Currently only 1 corpus is allowed.
    text=RETRIEVAL_QUERY,
    rag_retrieval_config=rag.RagRetrievalConfig(
        top_k=10,  # Optional
        filter=rag.Filter(
            vector_distance_threshold=0.5,  # Optional
        ),
    ),
)

# The retrieved context can be passed to any SDK or model generation API to generate final results.
retrieved_context = " ".join(
    [context.text for context in response.contexts.contexts]
).replace("\n", "")

retrieved_context

"""
## Cleaning up

Clean up resources created in this notebook.
"""

delete_rag_corpus = False  # @param {type:"boolean"}

if delete_rag_corpus:
    rag.delete_corpus(name=rag_corpus.name)



================================================
FILE: gemini/rag-engine/rag_engine_vertex_ai_search.ipynb
================================================
# Jupyter notebook converted to Python script.

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Vertex AI RAG Engine with Vertex AI Search

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_vertex_ai_search.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Frag-engine%2Frag_engine_vertex_ai_search.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/rag-engine/rag_engine_vertex_ai_search.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_vertex_ai_search.ipynb">
      <img width="32px" src="https://www.svgrepo.com/download/217753/github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>
"""

"""
| | |
|-|-|
| Author(s) | [Alex Dorozhkin](https://github.com/galexdor) |
"""

"""
## Overview

This notebook illustrates how to use [Vertex AI RAG Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview) with [Vertex AI Search](https://cloud.google.com/enterprise-search) as a retrieval backend. Vertex AI Search's ability to handle large datasets, provide low-latency retrieval, and improve scalability makes it a powerful tool for enhancing RAG applications.  By integrating Vertex AI Search, you can ensure that your RAG applications can efficiently access and process the necessary information for generating high-quality and contextually relevant responses.

For more information, refer to the [official documentation](https://cloud.google.com/generative-ai-app-builder/docs/enterprise-search-introduction).

For more details on RAG corpus/file management and detailed support please visit [Vertex AI RAG Engine API](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/rag-api)
"""

"""
## Get started
"""

"""
### Install Vertex AI SDK and other required packages

"""

%pip install --upgrade --user --quiet google-cloud-aiplatform google-cloud-discoveryengine

"""
### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.
"""

import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)

"""
<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>

"""

"""
### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).
"""

# Use the environment variable if the user doesn't provide Project ID.
import os

import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type:"string", isTemplate: true}
if PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)

"""
### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.
"""

import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user(project_id=PROJECT_ID)

"""
## (Optional) Setup Vertex AI Search Datastore and Engine
"""

"""
In this section, we have some helper methods to help you setup your Vertex AI Search. These methods handle the creation of resources like Data Stores and Engines, which can take a few minutes.

This section is not required if you already have a Vertex AI Search engine ready to use.

To get started using Vertex AI Search, you must have an existing Google Cloud project and [enable the Discovery Engine API](https://console.cloud.google.com/flows/enableapi?apiid=discoveryengine.googleapis.com).
"""

"""
### Initialize Vertex AI Search SDK
"""

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine

VERTEX_AI_SEARCH_LOCATION = "global"

"""
### Create and Populate a Datastore
"""

def create_data_store(
    project_id: str, location: str, data_store_name: str, data_store_id: str
):
    # Create a client
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )
    client = discoveryengine.DataStoreServiceClient(client_options=client_options)

    # Initialize request argument(s)
    data_store = discoveryengine.DataStore(
        display_name=data_store_name,
        industry_vertical=discoveryengine.IndustryVertical.GENERIC,
        content_config=discoveryengine.DataStore.ContentConfig.CONTENT_REQUIRED,
    )

    operation = client.create_data_store(
        request=discoveryengine.CreateDataStoreRequest(
            parent=client.collection_path(project_id, location, "default_collection"),
            data_store=data_store,
            data_store_id=data_store_id,
        )
    )

    # Make the request
    response = operation.result(timeout=90)
    return response.name

# The datastore name can only contain lowercase letters, numbers, and hyphens
DATASTORE_NAME = "alphabet-contracts"  # @param {type:"string", isTemplate: true}
DATASTORE_ID = f"{DATASTORE_NAME}-id"

create_data_store(PROJECT_ID, VERTEX_AI_SEARCH_LOCATION, DATASTORE_NAME, DATASTORE_ID)

def import_documents(
    project_id: str,
    location: str,
    data_store_id: str,
    gcs_uri: str,
):
    # Create a client
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )
    client = discoveryengine.DocumentServiceClient(client_options=client_options)

    # The full resource name of the search engine branch.
    # e.g. projects/{project}/locations/{location}/dataStores/{data_store_id}/branches/{branch}
    parent = client.branch_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        branch="default_branch",
    )

    source_documents = [f"{gcs_uri}/*"]

    request = discoveryengine.ImportDocumentsRequest(
        parent=parent,
        gcs_source=discoveryengine.GcsSource(
            input_uris=source_documents, data_schema="content"
        ),
        # Options: `FULL`, `INCREMENTAL`
        reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
    )

    # Make the request
    operation = client.import_documents(request=request)

    response = operation.result()

    # Once the operation is complete,
    # get information from operation metadata
    metadata = discoveryengine.ImportDocumentsMetadata(operation.metadata)

    # Handle the response
    return operation.operation.name

GCS_BUCKET = "gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs"  # @param {type:"string", isTemplate: true}

import_documents(PROJECT_ID, VERTEX_AI_SEARCH_LOCATION, DATASTORE_ID, GCS_BUCKET)

"""
### Create a Search Engine
"""

def create_engine(
    project_id: str, location: str, engine_name: str, engine_id: str, data_store_id: str
):
    # Create a client
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )
    client = discoveryengine.EngineServiceClient(client_options=client_options)

    # Initialize request argument(s)
    engine = discoveryengine.Engine(
        display_name=engine_name,
        solution_type=discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,
        industry_vertical=discoveryengine.IndustryVertical.GENERIC,
        data_store_ids=[data_store_id],
        search_engine_config=discoveryengine.Engine.SearchEngineConfig(
            search_tier=discoveryengine.SearchTier.SEARCH_TIER_ENTERPRISE,
        ),
    )

    request = discoveryengine.CreateEngineRequest(
        parent=client.collection_path(project_id, location, "default_collection"),
        engine=engine,
        engine_id=engine.display_name,
    )

    # Make the request
    operation = client.create_engine(request=request)
    response = operation.result(timeout=90)
    return response.name

ENGINE_NAME = DATASTORE_NAME
ENGINE_ID = DATASTORE_ID
create_engine(
    PROJECT_ID, VERTEX_AI_SEARCH_LOCATION, ENGINE_NAME, ENGINE_ID, DATASTORE_ID
)

"""
## Create a RAG corpus using Vertex AI Search Engine as the retrieval backend
"""

"""
### Import libraries
"""

from vertexai.preview import rag
from vertexai.preview.generative_models import GenerativeModel, Tool

"""
### Create RAG Config

"""

# Name your corpus
DISPLAY_NAME = ""  # @param {type:"string", "placeholder": "your-corpus-name"}

# Vertex AI Search name
ENGINE_NAME = ""  # @param {type:"string", "placeholder": "your-engine-name"}
vertex_ai_search_config = rag.VertexAiSearchConfig(
    serving_config=f"{ENGINE_NAME}/servingConfigs/default_search",
)

rag_corpus = rag.create_corpus(
    display_name=DISPLAY_NAME,
    vertex_ai_search_config=vertex_ai_search_config,
)

# Check the corpus just created
new_corpus = rag.get_corpus(name=rag_corpus.name)
new_corpus

"""
## Using Gemini GenerateContent API with Rag Retrieval Tool

"""

rag_resource = rag.RagResource(
    rag_corpus=rag_corpus.name,
)

rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag_resource],  # Currently only 1 corpus is allowed.
            similarity_top_k=10,
        ),
    )
)

rag_model = GenerativeModel("gemini-2.0-flash", tools=[rag_retrieval_tool])

"""
Note: The Vertex AI Search engine will take some time to be ready to query.

If you recently created an engine and you receive an error similar to:

`404 Engine {ENGINE_NAME} is not found`

Then wait a few minutes and try your query again.
"""

GENERATE_CONTENT_PROMPT = (
    "Who is CFO of Google?"  # @param {type:"string", isTemplate: true}
)

response = rag_model.generate_content(GENERATE_CONTENT_PROMPT)

response

"""
## Using other generation API with Rag Retrieval Tool

The retrieved contexts can be passed to any SDK or model generation API to generate final results.
"""

RETRIEVAL_QUERY = "Who is CFO of Google?"  # @param {type:"string", isTemplate: true}

rag_resource = rag.RagResource(rag_corpus=rag_corpus.name)

response = rag.retrieval_query(
    rag_resources=[rag_resource],  # Currently only 1 corpus is allowed.
    text=RETRIEVAL_QUERY,
    similarity_top_k=10,
)

# The retrieved context can be passed to any SDK or model generation API to generate final results.
retrieved_context = " ".join(
    [context.text for context in response.contexts.contexts]
).replace("\n", "")

retrieved_context

"""
## Cleaning up

Clean up RAG resources created in this notebook.
"""

delete_rag_corpus = False  # @param {type:"boolean"}

if delete_rag_corpus:
    rag.delete_corpus(name=rag_corpus.name)



================================================
FILE: gemini/rag-engine/rag_engine_weaviate.ipynb
================================================
# Jupyter notebook converted to Python script.

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Vertex AI RAG Engine with Weaviate

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_weaviate.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Frag-engine%2Frag_engine_weaviate.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/rag-engine/rag_engine_weaviate.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_weaviate.ipynb">
      <img width="32px" src="https://www.svgrepo.com/download/217753/github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

<div style="clear: both;"></div>

<b>Share to:</b>

<a href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_weaviate.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" alt="LinkedIn logo">
</a>

<a href="https://bsky.app/intent/compose?text=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_weaviate.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/7/7a/Bluesky_Logo.svg" alt="Bluesky logo">
</a>

<a href="https://twitter.com/intent/tweet?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_weaviate.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/5a/X_icon_2.svg" alt="X logo">
</a>

<a href="https://reddit.com/submit?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_weaviate.ipynb" target="_blank">
  <img width="20px" src="https://redditinc.com/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" alt="Reddit logo">
</a>

<a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_weaviate.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook logo">
</a>            
"""

"""
| | |
|-|-|
| Author(s) | [Ming Zhang](https://github.com/mzhang-ai) |
"""

"""
## Overview

This notebook illustrates how to use [Vertex AI RAG Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview) with [Weaviate](https://weaviate.io/) as a vector database.

For more information, refer to the [official documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/use-weaviate-db).

For more details on RAG corpus/file management and detailed support please visit [Vertex AI RAG Engine API](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/rag-api)
"""

"""
## Get started
"""

"""
### Install Vertex AI SDK and other required packages

"""

%pip install --upgrade --user --quiet google-cloud-aiplatform

"""
### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.
"""

import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)

"""
<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>

"""

"""
### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.
"""

import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()

"""
### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).
"""

# Use the environment variable if the user doesn't provide Project ID.
import os

import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type:"string", isTemplate: true}
if PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)

"""
## Create a RAG corpus using Weaviate as the Vector Database
"""

"""
### Import libraries
"""

from vertexai.preview import rag
from vertexai.preview.generative_models import GenerativeModel, Tool

"""
### Load embedding model and create RAG Config
"""

# Configure a Google first-party embedding model
embedding_model_config = rag.EmbeddingModelConfig(
    publisher_model="publishers/google/models/text-embedding-005"
)

# Name your corpus
DISPLAY_NAME = ""  # @param {type:"string", "placeholder": "your-corpus-name"}

# Configure a Weaviate Vector Database Instance for the corpus
# More details for how to deploy a Weaviate Database Instance
# https://cloud.google.com/vertex-ai/generative-ai/docs/use-weaviate-db
WEAVIATE_HTTP_ENDPOINT = (
    ""  # @param {type:"string", "placeholder": "your-weaviate-http-endpoint"}
)
COLLECTION_NAME = (
    ""  # @param {type:"string", "placeholder": "your-weaviate-collection-name"}
)
API_KEY = (
    ""  # @param {type:"string", "placeholder": "your-secret-manager-resource-name"}
)
vector_db = rag.Weaviate(
    weaviate_http_endpoint=WEAVIATE_HTTP_ENDPOINT,
    collection_name=COLLECTION_NAME,
    api_key=API_KEY,
)

rag_corpus = rag.create_corpus(
    display_name=DISPLAY_NAME,
    embedding_model_config=embedding_model_config,
    vector_db=vector_db,
)

# Check the corpus just created
new_corpus = rag.get_corpus(name=rag_corpus.name)
new_corpus

"""
## Upload a file to the corpus
"""

%%writefile test.txt

Here's a demo for Weaviate RAG.

rag_file = rag.upload_file(
    corpus_name=rag_corpus.name,
    path="test.txt",
    display_name="test.txt",
    description="my test",
)

"""
## Import files from Google Cloud Storage

Remember to grant "Viewer" access to the "Vertex RAG Data Service Agent" (with the format of `service-{project_number}@gcp-sa-vertex-rag.iam.gserviceaccount.com`) for your Google Cloud Storage bucket
"""

GCS_BUCKET = ""  # @param {type:"string", "placeholder": "your-gs-bucket"}

response = rag.import_files(  # noqa: F704
    corpus_name=rag_corpus.name,
    paths=[GCS_BUCKET],
    chunk_size=512,
    chunk_overlap=50,
)

# Check the files just imported. It may take a few seconds to process the imported files.
rag.list_files(corpus_name=rag_corpus.name)

"""
## Import files from Google Drive

Eligible paths can be:

- `https://drive.google.com/drive/folders/{folder_id}`
- `https://drive.google.com/file/d/{file_id}`

Remember to grant "Viewer" access to the "Vertex RAG Data Service Agent" (with the format of `service-{project_number}@gcp-sa-vertex-rag.iam.gserviceaccount.com`) for your Drive folder/files.

"""

FILE_ID = ""  # @param {type:"string", "placeholder": "your-file-id"}
FILE_PATH = f"https://drive.google.com/file/d/{FILE_ID}"

rag.import_files(
    corpus_name=rag_corpus.name,
    paths=[FILE_PATH],
    chunk_size=1024,
    chunk_overlap=100,
)

# Check the files just imported. It may take a few seconds to process the imported files.
rag.list_files(corpus_name=rag_corpus.name)

"""
## Using Gemini GenerateContent API with Rag Retrieval Tool

When retrieved contexts similarity distance < `vector_distance_threshold`, the contexts (from `RagStore`) will be used for content generation.
"""

rag_resource = rag.RagResource(
    rag_corpus=rag_corpus.name,
)

rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag_resource],  # Currently only 1 corpus is allowed.
            similarity_top_k=10,
            vector_distance_threshold=0.4,
        ),
    )
)

rag_model = GenerativeModel("gemini-2.0-flash", tools=[rag_retrieval_tool])

GENERATE_CONTENT_PROMPT = "What is RAG and why it is helpful?"  # @param {type:"string"}

response = rag_model.generate_content(GENERATE_CONTENT_PROMPT)

response

"""
## Using other generation API with Rag Retrieval Tool

The retrieved contexts can be passed to any SDK or model generation API to generate final results.
"""

RETRIEVAL_QUERY = "What is RAG and why it is helpful?"  # @param {type:"string"}

rag_resource = rag.RagResource(
    rag_corpus=rag_corpus.name,
    # Need to manually get the ids from rag.list_files.
    # rag_file_ids=[],
)

response = rag.retrieval_query(
    rag_resources=[rag_resource],  # Currently only 1 corpus is allowed.
    text=RETRIEVAL_QUERY,
    similarity_top_k=10,
)

# The retrieved context can be passed to any SDK or model generation API to generate final results.
retrieved_context = " ".join(
    [context.text for context in response.contexts.contexts]
).replace("\n", "")

retrieved_context

"""
## Cleaning up

Clean up resources created in this notebook.
"""

delete_rag_corpus = False  # @param {type:"boolean"}

if delete_rag_corpus:
    rag.delete_corpus(name=rag_corpus.name)


