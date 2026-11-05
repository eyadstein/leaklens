# MCP server (optional, Python 3.11+)

LeakLens can expose its Python/notebook scanner to MCP-compatible AI assistants. The server is read-only and runs locally over **stdio**; it never executes scanned Python code.

```bash
pip install "leaklens-ml[mcp]"
python -m leaklens.mcp_server
```

An MCP host (for example, an editor that supports local MCP servers) should launch the server as a subprocess. Example host config:

```json
{
  "mcpServers": {
    "leaklens": {
      "command": "python",
      "args": ["-m", "leaklens.mcp_server"],
      "env": {"LEAKLENS_MCP_ROOT": "/path/to/your/project"}
    }
  }
}
```

Tools:
- `scan_python_code(code, filename="input.py")` analyzes pasted Python snippets up to 512 KiB.
- `scan_local_file(path)` analyzes one `.py` or `.ipynb` within `LEAKLENS_MCP_ROOT`, maximum 2 MiB per file. Symlinks cannot escape the root. If unset, the root is the server's current working directory.

Both tools return trust score, finding count and structured LeakLens findings. Paths and source text supplied by a model are **untrusted**; users should choose an explicit scan root, review findings, and avoid sending sensitive code to external models.

For Windows, use a path such as `C:\\Users\\you\\project` in the JSON configuration (backslashes must be escaped). On hosts that support only shell commands, use `leaklens-mcp`.

The `mcp` dependency is optional, so normal LeakLens CLI and CI installations do not need the MCP SDK. The server communicates via standard input/output, not a network listener. It only scans input and never executes it.
