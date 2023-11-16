# FastAPI Gateway to GitHub API Fabric

This repository hosts a FastAPI Coding Co-pilot  designed as an interface with GitHub's search API. It facilitates a range of search operations including issues, commits, code, users, topics, and repositories. The application strikes a balance by offering a scalable, extensible, and robust infrastructure, tailored to enhance AI applications. It's adept at managing intricate data processing tasks and adeptly delivering intelligent functionalities with efficiency.

## Features

- OAuth2 and Bearer token based security for endpoints.
- GitHub API integration to search for different entities (code, commits, issues, labels, repositories, topics, users).
- Asynchronous HTTP requests for improved performance.
- Paginated responses for large result sets.
- CORS middleware for cross-origin resource sharing.
- Automatic redirection to FastAPI's auto-generated interactive API documentation.

# API Fabric for AI Applications

The structure provided by `main.py` using FastAPI serves as a core element for an API fabric, initially built on the GitHub Search API. This fabric underpins AI applications such as ChatGPT with GPT-3 assistants by enabling efficient, scalable, and secure interactions with diverse data sources and services.

## Modular Design and API Compatibility

FastAPI's modular nature allows for straightforward expansion and management of endpoints. While the current structure is tied to GitHub's Search API, it could easily be modified to integrate with other APIs such as Stripe for payments, Microsoft Graph for Office 365 services, or Google's various APIs.

## Asynchronous Processing

The asynchronous processing capabilities of FastAPI benefit AI application requests that may take longer to process. This functionality is key to improved concurrency, reduced response times, and increased system throughput.

## Security and Access Control

The existing security implementations demonstrate how application access can be controlled and secured—a key aspect for handling sensitive AI-generated data or proprietary algorithms. The platform is designed to support advanced OAuth2 flows, enhancing security as applications evolve.

## Extensibility and Integration

Developers can extend this API fabric to integrate additional APIs, orchestrate complex data workflows, and standardize communication protocols. This facilitates efficient AI functions like natural language understanding, response generation, and user interaction handling typical of GPT-3 assistants.

## Real-time Processing and Task Management

With support for background tasks and real-time processing, the FastAPI infrastructure is well-suited for high-demand and real-time AI interactions, such as streaming data analysis and conversational agents, catering to the needs of dynamic AI applications.

## Prerequisites

To run this API gateway, you will need:

- Python 3.6+
- Packages: `fastapi`, `httpx`, `uvicorn`, `pydantic`, `python-multipart`, `python-jose`, `passlib`, `bcrypt`

## Environment Variables

Before running the server, ensure the following environment variables are set:

- `FASTAPI_SERVER_URL`: The server URL where FastAPI will serve the API (optional).
- `GITHUB_TOKEN`: Personal GitHub token for API access.
- `API_BEARER_TOKEN`: Secret token that clients should use to access the API.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ruvnet/coding-wingman.git
cd coding-wingman
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```
## #Usage
To start the server, run:
```
uvicorn main:app --reload
```
Visit http://localhost:8000/docs in your web browser to see the interactive API documentation provided by FastAPI.

# Endpoints Overview
The following endpoints are available in the API:

- `POST /search/code/`: Search for code snippets on GitHub.
- `POST /search/commits/`: Search for commits on GitHub.
- `POST /search/issues/`: Search for issues on GitHub, with automatic pagination.
- `POST /search/labels/`: Search for labels on GitHub.
- `POST /search/repositories/`: Search for repositories on GitHub.
- `POST /search/topics/`: Search for topics on GitHub.
- `POST /search/users/`: Search for users on GitHub.

To access the endpoints, provide a Bearer token in the Authorization header of your request.

## Security
This application uses the HTTPBearer security scheme. Ensure that only authorized clients can access the API by setting up the `API_BEARER_TOKEN` environment variable.

## Prerequisites
- Python 3.6+
- FastAPI
- Uvicorn
- HTTPX
- Pydantic

## Installation
To install the required packages, run the following command:

```bash
pip install fastapi uvicorn httpx pydantic
```
To set the environment variables for local development, you can use the export command (on Unix-based systems) or set command (on Windows):
```
export GITHUB_TOKEN='your_github_token'
export API_BEARER_TOKEN='your_api_bearer_token'
export FASTAPI_SERVER_URL='your_fastapi_server_url' # Optional
```
Alternatively, you can use a .env file and load it with a library such as python-dotenv.

Running the Application
To start the application, run the following command:

```
uvicorn main:app --host 0.0.0.0 --port 8000
```
The application will be available at the URL specified by the FASTAPI_SERVER_URL environment variable, or by default at http://localhost:8000.

