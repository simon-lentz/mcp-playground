# mcp-playground

## Background

### Static Models

At present, the general expectations for interaction with a generative model are high reponse fluency and accuracy. The utility associated with
the ability to prompt on an arbitrary topic and receive relevant information has driven the development of LLMs to the present SOTA. Numerous strategies
exist to improve the accuracy, relevance, etc. of the generated content: retrieval augmented generation, fine-tuning, and chain of thought reasoning represent
a few of recent strategies highlighted in research work and 'production' integration of static models.

By design, a 'static' model lacks the agency needed to execute on the content it generates. Presently, that is the responsibility of the user.
How can we move from a passive static model to something with greater capacity to *do*?

### Dynamic Agents and the MCP

Enter the agentic model. In November 2024, Anthropic introduced the [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol). MCP is,
in Anthropic's words, "an open protocol that enables seamless integration between LLM applications and external data sources and tools." MCP leverages
the familar client-server architecture to standardize the communication between the generative model and the outside world.

* **Client** - traditionally, this would be a web browser or something similar that resides on a user's computer and serves requests on their behalf.
In the MCP context, the client serves as a two-way street between the AI model and the resources that we make available for request to the AI model.

* **Server** - in the internet client-server model a server typically stores files until they are requested by a client, at which point it sends a response
containing the requested content. The MCP server is the substrate on which we can define resources in a common-language that our AI model(s) can utilize.
In this repo, the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) allows for the creation of tools, resources, and prompts that
can be consumed by AI model.

#### Tools

Tools are for **action**. They are executable functions that the agent calls to interact with the world.

#### Prompts

Prompts are for **guidance**. They are user-initiated templates that orchestrate a complex reasoning workflow for the LLM.

#### Resources

Resources are for **context**. They are passive, read-only data sources the agent consults to become informed before acting.

## Usage

### `src/single-server`

Copy the environment template file into a local .env file, source that file:

```bash
cp .env.tmpl .env
source .env
```

Run the mcp client for the single-server demonstrator using python:

```bash
python src/single-server/mcp_client.py
```

You should expect to see an output like:

```bash
[07/24/25 13:29:19] INFO     Processing request of type ListToolsRequest                                                                                                                                                           server.py:625
Weather MCP agent is ready.
Type a question, or use one of the following commands:
  /prompts                              - to list available prompts
  /prompt    <prompt_name> "args"...  - to run a specific prompt
  /resources                            - to list available resources
  /resource  <resource_uri>           - to load a resource for the agent
```
