#!/usr/bin/env python3
"""GigPilot: an agent that turns a client's job post into a priced, ready-to-send proposal.

Runs on Nebius Token Factory with an NVIDIA Nemotron model (OpenAI-compatible tool calling).
Python 3.9+, standard library only.
"""

import ast
import ipaddress
import json
import operator as op
import os
import re
import socket
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/")
MODEL = os.getenv("NEBIUS_MODEL", "nvidia/nemotron-3-super-120b-a12b")
KEY = os.getenv("NEBIUS_API_KEY", "")
TAVILY = os.getenv("TAVILY_API_KEY", "")
OUT = Path("proposals")
MAX_STEPS = 8

SYSTEM = """You are GigPilot, an agent that helps freelancers win jobs. You get a client's job post and the freelancer's hourly rate. Work step by step with tools:
1. If the post contains a URL, call fetch_url on it.
2. If web_search is available, use it to sanity-check market rates or the client's domain.
3. Use calculator for every number in the quote (hours x rate, rush fees, totals). Never do arithmetic in your head.
4. Call save_proposal exactly once with the finished proposal in Markdown: short intro, your understanding of the need, scope, timeline, itemised price, next step.
Then reply with a 2-sentence summary for the freelancer. Be concrete, warm and concise."""


def post(url, body, headers):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    try:
        return json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url.split('/')[2]} returned {e.code}: {e.read().decode()[:300]}")


# ---- tools -----------------------------------------------------------------
OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg}


def _eval(n):
    if isinstance(n, ast.Expression):
        return _eval(n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
        return OPS[type(n.op)](_eval(n.operand))
    if isinstance(n, ast.BinOp) and type(n.op) in OPS:
        a, b = _eval(n.left), _eval(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > 10:
            raise ValueError("exponent too large")
        return OPS[type(n.op)](a, b)
    raise ValueError("unsupported expression")


def calculator(a):
    return str(round(_eval(ast.parse(a["expression"], mode="eval")), 2))


def fetch_url(a):
    u = urlparse(a["url"])
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("only http(s) URLs are allowed")
    for *_, sock in socket.getaddrinfo(u.hostname, None):
        if ipaddress.ip_address(sock[0]).is_private or ipaddress.ip_address(sock[0]).is_loopback:
            raise ValueError("private addresses are blocked")
    req = urllib.request.Request(a["url"], headers={"User-Agent": "GigPilot/1.0"})
    html = urllib.request.urlopen(req, timeout=15).read(300_000).decode("utf-8", "ignore")
    text = re.sub(r"<(script|style)[\s\S]*?</\1>|<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()[:4000]


def web_search(a):
    r = post("https://api.tavily.com/search", {"query": a["query"], "max_results": 4, "include_answer": True},
             {"Authorization": "Bearer " + TAVILY})
    hits = "\n".join(f"- {x['title']}: {x['content'][:200]}" for x in r.get("results", []))
    return f"{r.get('answer', '')}\n{hits}".strip()


def save_proposal(a):
    OUT.mkdir(exist_ok=True)
    name = re.sub(r"[^a-z0-9]+", "-", a["title"].lower()).strip("-")[:50] or "proposal"
    (OUT / f"{name}.md").write_text(a["markdown"], encoding="utf-8")
    return f"saved proposals/{name}.md"


def tool(name, desc, **props):
    schema = {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()}, "required": list(props)}
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}


TOOLS = [
    tool("fetch_url", "Read a web page, e.g. a link inside the job post.", url="Full http(s) URL"),
    tool("web_search", "Search the web for market rates or client background.", query="Search query"),
    tool("calculator", "Evaluate arithmetic such as 12*35*1.2.", expression="Arithmetic expression"),
    tool("save_proposal", "Save the final proposal as a Markdown file.", title="Short title", markdown="Full proposal in Markdown"),
]
FUNCS = {"fetch_url": fetch_url, "web_search": web_search, "calculator": calculator, "save_proposal": save_proposal}


# ---- agent loop ------------------------------------------------------------
def agent(brief, rate, cur):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Hourly rate: {rate} {cur}\n\nClient job post:\n{brief}"}]
    tools = TOOLS if TAVILY else [t for t in TOOLS if t["function"]["name"] != "web_search"]
    for _ in range(MAX_STEPS):
        r = post(f"{BASE}/chat/completions", {"model": MODEL, "messages": msgs, "tools": tools, "tool_choice": "auto",
                                              "temperature": 0.3, "max_tokens": 2500}, {"Authorization": "Bearer " + KEY})
        m = r["choices"][0]["message"]
        calls, text = m.get("tool_calls") or [], m.get("content") or ""
        msgs.append({"role": "assistant", "content": text, **({"tool_calls": calls} if calls else {})})
        if not calls:
            yield {"type": "final", "text": re.sub(r"<think>[\s\S]*?</think>", "", text).strip()}
            return
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
                yield {"type": "call", "tool": name, "args": args}
                out = FUNCS[name](args)
            except Exception as e:
                out = f"error: {e}"
            yield {"type": "result", "tool": name, "text": out[:600]}
            if name == "save_proposal" and not out.startswith("error"):
                yield {"type": "proposal", "markdown": args["markdown"]}
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
    yield {"type": "final", "text": "Stopped: the agent reached its step limit."}


# ---- web UI ----------------------------------------------------------------
PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>GigPilot</title><style>
:root{--rail:#0F3D3A;--ink:#12211F;--paper:#F5F6F2;--sea:#1B8A83;--amber:#E0A526;--mut:#5B6B68}
*{box-sizing:border-box}body{margin:0;font:16px/1.55 system-ui,sans-serif;color:var(--ink);background:var(--paper);display:grid;grid-template-columns:minmax(300px,380px) 1fr;min-height:100vh}
aside{background:var(--rail);color:#E8F1EF;padding:32px 28px;display:flex;flex-direction:column;gap:14px}
h1{font:700 34px/1.1 Georgia,serif;margin:0}aside p{margin:0;color:#A9C7C2;font-size:14px}label{font-size:14px;color:#A9C7C2}
textarea,input,select{width:100%;font:inherit;padding:10px 12px;border:1px solid #2F6660;border-radius:6px;background:#0B302D;color:#fff}
textarea{min-height:220px;resize:vertical}.row{display:grid;grid-template-columns:1fr 110px;gap:10px}
button{font:600 16px system-ui;padding:12px;border:0;border-radius:6px;background:var(--amber);color:#2A1E00;cursor:pointer}button:disabled{opacity:.6;cursor:wait}
main{padding:32px clamp(20px,4vw,56px);max-width:820px}h2{font:700 22px Georgia,serif;margin:0 0 12px}
#trace{list-style:none;margin:0 0 32px;padding:0 0 0 18px;border-left:2px solid #C9D6D3}
#trace:empty::before{content:"Paste a job post and the agent's steps will appear here.";color:var(--mut)}
#trace li{margin:0 0 10px;font-size:15px;position:relative}
#trace li::before{content:"";position:absolute;left:-25px;top:8px;width:10px;height:10px;border-radius:50%;background:var(--sea)}
li.call::before{background:var(--amber)}li.err{color:#B3261E}li.final{font-weight:600}li.res{color:var(--mut);font-size:14px}
code{font:13px ui-monospace,monospace;background:#E4EAE7;padding:2px 6px;border-radius:4px;word-break:break-all}
#sheet{background:#fff;border:1px solid #D5DEDB;padding:24px 28px}#doc{white-space:pre-wrap;font:16px/1.65 Georgia,serif;margin:0 0 16px}
#copy{background:var(--sea);color:#fff;padding:10px 18px}:focus-visible{outline:3px solid var(--amber);outline-offset:2px}
@media(max-width:800px){body{grid-template-columns:1fr}}
</style></head><body>
<aside><h1>GigPilot</h1><p>Paste a client's job post. The agent reads links, checks rates, does the maths and saves a ready-to-send proposal.</p>
<label for=brief>Client job post</label><textarea id=brief placeholder="Need a Shopify store redesign, 5 pages, live in 2 weeks. Reference: https://example.com"></textarea>
<div class=row><div><label for=rate>Your hourly rate</label><input id=rate type=number value=25 min=1></div>
<div><label for=cur>Currency</label><select id=cur><option>USD<option>PKR<option>EUR<option>GBP</select></div></div>
<button id=go>Draft proposal</button><p>Model: __MODEL__ on Nebius Token Factory</p></aside>
<main><h2>Agent trace</h2><ol id=trace aria-live=polite></ol>
<section id=sheet hidden><h2>Proposal</h2><pre id=doc></pre><button id=copy>Copy proposal</button></section></main>
<script>
const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const add=(cls,html)=>{const li=document.createElement('li');li.className=cls;li.innerHTML=html;$('#trace').append(li)};
$('#go').onclick=async()=>{
  const brief=$('#brief').value.trim();if(!brief){$('#brief').focus();return}
  $('#trace').innerHTML='';$('#sheet').hidden=true;$('#go').disabled=true;$('#go').textContent='Agent working…';
  try{
    const r=await fetch('/api/run',{method:'POST',body:JSON.stringify({brief,rate:$('#rate').value,currency:$('#cur').value})});
    const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
    for(;;){const {done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\n');buf=lines.pop();
      for(const l of lines){if(!l)continue;const e=JSON.parse(l);
        if(e.type==='call')add('call',`<b>${e.tool}</b> <code>${esc(JSON.stringify(e.args).slice(0,160))}</code>`);
        else if(e.type==='result')add('res',esc(e.text));
        else if(e.type==='proposal'){$('#doc').textContent=e.markdown;$('#sheet').hidden=false}
        else if(e.type==='final')add('final',esc(e.text));
        else if(e.type==='error')add('err',esc(e.text))}}
  }catch(x){add('err',esc(x.message))}
  $('#go').disabled=false;$('#go').textContent='Draft proposal';
};
$('#copy').onclick=()=>navigator.clipboard.writeText($('#doc').textContent);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE.replace("__MODEL__", MODEL).encode())

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if not KEY: raise RuntimeError("Set NEBIUS_API_KEY before starting the server (see README).")
            for e in agent(d["brief"], d.get("rate", "25"), d.get("currency", "USD")):
                self.wfile.write((json.dumps(e) + "\n").encode())
                self.wfile.flush()
        except Exception as e:
            self.wfile.write((json.dumps({"type": "error", "text": str(e)}) + "\n").encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"GigPilot running at http://127.0.0.1:{port}  (model: {MODEL})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
