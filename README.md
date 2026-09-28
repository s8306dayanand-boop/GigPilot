# GigPilot

GigPilot is a lightweight AI agent that turns a client job post into a professional freelance proposal.

It uses:
- Nebius Token Factory for model access
- NVIDIA Nemotron as the default reasoning model
- Optional Tavily search for market rates and client sanity checks
- A safe in-app calculator using Python `ast` instead of `eval`

## What it does

1. Reads the client post and the freelancer's hourly rate.
2. Opens public URLs found in the post.
3. Optionally checks market rates or client background with web search.
4. Calculates the quote line by line.
5. Saves the final proposal as a Markdown file in `proposals/`.
6. Streams the agent trace live in the browser.

## Requirements

- Python 3.9+
- A Nebius Token Factory API key
- Optional Tavily API key to enable web search

## Environment variables

```bash
export NEBIUS_API_KEY="your-token-factory-key"
export NEBIUS_MODEL="nvidia/nemotron-3-super-120b-a12b"
export NEBIUS_BASE_URL="https://api.tokenfactory.nebius.com/v1"
export TAVILY_API_KEY="optional-tavily-key"
export PORT="8000"
```

## Run locally

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

Paste a job post, set your hourly rate, and click Draft proposal.

## Safety notes

- `fetch_url` allows only public `http(s)` links.
- Local/private IPs are blocked.
- `calculator` evaluates arithmetic with Python `ast` for safe parsing.
- The app binds only to `127.0.0.1`.

## License

MIT
