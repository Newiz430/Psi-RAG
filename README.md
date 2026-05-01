# ᛉ-RAG: Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation

<div align="center">
<img src="fig/psi-algiz-rune.png" style="zoom: 15%;" />
<br />
<span style="color: #666666;">Artwork by </span>
<a href="https://thorolfw.artstation.com/" style="color: #0366d6;">Thorolf Wolfson</a>
<span style="color: #666666;">.</span>
</div>

<br>

<div align="center">

[![Github](https://img.shields.io/badge/GitHub-repo-blue?style=for-the-badge&logo=github)](https://github.com/Newiz430/Psi-RAG)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-repo-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/Newiz430/Psi-RAG)
[![Paper](https://img.shields.io/badge/Paper-Soon-red?style=for-the-badge&logo=arxiv&logoColor=white)](TBD)
[![Paper](https://img.shields.io/badge/Paper-Soon-24282c?style=for-the-badge&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjxzdmcKICAgaW5rc2NhcGU6dmVyc2lvbj0iMS4yLjIgKGIwYTg0ODY1LCAyMDIyLTEyLTAxKSIKICAgc29kaXBvZGk6ZG9jbmFtZT0iaWNtbC1uYXZiYXItbG9nby5zdmciCiAgIHZpZXdCb3g9IjAgMCA5OS4yNjQgOTcuNjY1MDAxIgogICBoZWlnaHQ9Ijk3LjY2NTAwMSIKICAgd2lkdGg9Ijk5LjI2NCIKICAgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIKICAgaWQ9InN2ZzYwNCIKICAgdmVyc2lvbj0iMS4xIgogICB4bWxuczppbmtzY2FwZT0iaHR0cDovL3d3dy5pbmtzY2FwZS5vcmcvbmFtZXNwYWNlcy9pbmtzY2FwZSIKICAgeG1sbnM6c29kaXBvZGk9Imh0dHA6Ly9zb2RpcG9kaS5zb3VyY2Vmb3JnZS5uZXQvRFREL3NvZGlwb2RpLTAuZHRkIgogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyIKICAgeG1sbnM6Y2M9Imh0dHA6Ly9jcmVhdGl2ZWNvbW1vbnMub3JnL25zIyIKICAgeG1sbnM6ZGM9Imh0dHA6Ly9wdXJsLm9yZy9kYy9lbGVtZW50cy8xLjEvIj48bWV0YWRhdGEKICAgICBpZD0ibWV0YWRhdGE2MTAiPjxyZGY6UkRGPjxjYzpXb3JrCiAgICAgICAgIHJkZjphYm91dD0iIj48ZGM6Zm9ybWF0PmltYWdlL3N2Zyt4bWw8L2RjOmZvcm1hdD48ZGM6dHlwZQogICAgICAgICAgIHJkZjpyZXNvdXJjZT0iaHR0cDovL3B1cmwub3JnL2RjL2RjbWl0eXBlL1N0aWxsSW1hZ2UiIC8+PC9jYzpXb3JrPjwvcmRmOlJERj48L21ldGFkYXRhPjxkZWZzCiAgICAgaWQ9ImRlZnM2MDgiPjxjbGlwUGF0aAogICAgICAgaWQ9ImNsaXBQYXRoNjI0IgogICAgICAgY2xpcFBhdGhVbml0cz0idXNlclNwYWNlT25Vc2UiPjxwYXRoCiAgICAgICAgIGlkPSJwYXRoNjIyIgogICAgICAgICBkPSJNIDAsMTA4IEggMjE2IFYgMCBIIDAgWiIgLz48L2NsaXBQYXRoPjwvZGVmcz48c29kaXBvZGk6bmFtZWR2aWV3CiAgICAgaW5rc2NhcGU6Y3VycmVudC1sYXllcj0iZzgwOCIKICAgICBpbmtzY2FwZTp3aW5kb3ctbWF4aW1pemVkPSIwIgogICAgIGlua3NjYXBlOndpbmRvdy15PSI0NzUiCiAgICAgaW5rc2NhcGU6d2luZG93LXg9IjExMDkiCiAgICAgaW5rc2NhcGU6Y3k9IjQ2LjI0OCIKICAgICBpbmtzY2FwZTpjeD0iMzMuNDgxMDkxIgogICAgIGlua3NjYXBlOnpvb209IjkuNzUxNzczMSIKICAgICBmaXQtbWFyZ2luLWJvdHRvbT0iNCIKICAgICBmaXQtbWFyZ2luLXJpZ2h0PSI0IgogICAgIGZpdC1tYXJnaW4tbGVmdD0iNCIKICAgICBmaXQtbWFyZ2luLXRvcD0iNCIKICAgICBsb2NrLW1hcmdpbnM9InRydWUiCiAgICAgc2hvd2dyaWQ9ImZhbHNlIgogICAgIGlkPSJuYW1lZHZpZXc2MDYiCiAgICAgaW5rc2NhcGU6d2luZG93LWhlaWdodD0iOTI2IgogICAgIGlua3NjYXBlOndpbmRvdy13aWR0aD0iMTY5MiIKICAgICBpbmtzY2FwZTpwYWdlc2hhZG93PSIyIgogICAgIGlua3NjYXBlOnBhZ2VvcGFjaXR5PSIwIgogICAgIGd1aWRldG9sZXJhbmNlPSIxMCIKICAgICBncmlkdG9sZXJhbmNlPSIxMCIKICAgICBvYmplY3R0b2xlcmFuY2U9IjEwIgogICAgIGJvcmRlcm9wYWNpdHk9IjEiCiAgICAgYm9yZGVyY29sb3I9IiM2NjY2NjYiCiAgICAgcGFnZWNvbG9yPSIjZmZmZmZmIgogICAgIGlua3NjYXBlOnNob3dwYWdlc2hhZG93PSIyIgogICAgIGlua3NjYXBlOnBhZ2VjaGVja2VyYm9hcmQ9IjAiCiAgICAgaW5rc2NhcGU6ZGVza2NvbG9yPSIjZDFkMWQxIiAvPjxnCiAgICAgdHJhbnNmb3JtPSJtYXRyaXgoMS4zMzMzMzMzLDAsMCwtMS4zMzMzMzMzLC05LjQwMzczMzEsMTE1LjU1ODQpIgogICAgIGlua3NjYXBlOmxhYmVsPSJJQ01MIExvZ28gMjAxOCIKICAgICBpbmtzY2FwZTpncm91cG1vZGU9ImxheWVyIgogICAgIGlkPSJnNjEyIj48ZwogICAgICAgaWQ9Imc2MTgiPjxyZWN0CiAgICAgICAgIHN0eWxlPSJmaWxsOiMyMzFmMjA7ZmlsbC1vcGFjaXR5OjE7c3Ryb2tlOiNmZmZmZmY7c3Ryb2tlLXdpZHRoOjA7c3Ryb2tlLWRhc2hhcnJheTpub25lIgogICAgICAgICBpZD0icmVjdDEzODQiCiAgICAgICAgIHdpZHRoPSI4LjYxMzgxODIiCiAgICAgICAgIGhlaWdodD0iOC44NDQ1NDU0IgogICAgICAgICB4PSI1MS4xMjE3MDgiCiAgICAgICAgIHk9Ii03Ny45MDExNjEiCiAgICAgICAgIHRyYW5zZm9ybT0ic2NhbGUoMSwtMSkiIC8+PHJlY3QKICAgICAgICAgc3R5bGU9ImZpbGw6IzIzMWYyMDtmaWxsLW9wYWNpdHk6MTtzdHJva2U6I2ZmZmZmZjtzdHJva2Utd2lkdGg6MDtzdHJva2UtZGFzaGFycmF5Om5vbmUiCiAgICAgICAgIGlkPSJyZWN0MTM4NC01IgogICAgICAgICB3aWR0aD0iNy43Njc4MTg1IgogICAgICAgICBoZWlnaHQ9IjcuNTM3MDkwOCIKICAgICAgICAgeD0iNTUuODkwMDcyIgogICAgICAgICB5PSItNjMuNjM0NTI1IgogICAgICAgICB0cmFuc2Zvcm09InNjYWxlKDEsLTEpIiAvPjxyZWN0CiAgICAgICAgIHN0eWxlPSJmaWxsOiMyMzFmMjA7ZmlsbC1vcGFjaXR5OjE7c3Ryb2tlOiNmZmZmZmY7c3Ryb2tlLXdpZHRoOjA7c3Ryb2tlLWRhc2hhcnJheTpub25lIgogICAgICAgICBpZD0icmVjdDEzODQtNS05IgogICAgICAgICB3aWR0aD0iNC42OTE0NTU0IgogICAgICAgICBoZWlnaHQ9IjQuNjE0NTQ1OCIKICAgICAgICAgeD0iNjAuNTQzMDcyIgogICAgICAgICB5PSItNzAuNjcxNzA3IgogICAgICAgICB0cmFuc2Zvcm09InNjYWxlKDEsLTEpIiAvPjxyZWN0CiAgICAgICAgIHN0eWxlPSJmaWxsOiMyMzFmMjA7ZmlsbC1vcGFjaXR5OjE7c3Ryb2tlOiNmZmZmZmY7c3Ryb2tlLXdpZHRoOjA7c3Ryb2tlLWRhc2hhcnJheTpub25lIgogICAgICAgICBpZD0icmVjdDEzODQtOSIKICAgICAgICAgd2lkdGg9IjEyLjIyODU0NSIKICAgICAgICAgaGVpZ2h0PSIxMi42OSIKICAgICAgICAgeD0iNDMuMjAwMDczIgogICAgICAgICB5PSItNTcuMjUxMDcyIgogICAgICAgICB0cmFuc2Zvcm09InNjYWxlKDEsLTEpIiAvPjxnCiAgICAgICAgIGNsaXAtcGF0aD0idXJsKCNjbGlwUGF0aDYyNCkiCiAgICAgICAgIGlkPSJnNjIwIj48ZwogICAgICAgICAgIGlkPSJnNTExIgogICAgICAgICAgIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQuNSwtNC41KSI+PGcKICAgICAgICAgICAgIHRyYW5zZm9ybT0idHJhbnNsYXRlKDI1LjIyNTYsNTcuOTcyNykiCiAgICAgICAgICAgICBpZD0iZzc5MiI+PHBhdGgKICAgICAgICAgICAgICAgaWQ9InBhdGg3OTQiCiAgICAgICAgICAgICAgIHN0eWxlPSJmaWxsOiM5NGQyMjI7ZmlsbC1vcGFjaXR5OjE7ZmlsbC1ydWxlOm5vbnplcm87c3Ryb2tlOm5vbmUiCiAgICAgICAgICAgICAgIGQ9Im0gMCwwIGMgMCwtMC42ODIgLTAuNTUzLC0xLjIzNSAtMS4yMzUsLTEuMjM1IGggLTguNCBjIC0wLjY4MiwwIC0xLjIzNiwwLjU1MyAtMS4yMzYsMS4yMzUgdiA4LjQgYyAwLDAuNjgzIDAuNTU0LDEuMjM2IDEuMjM2LDEuMjM2IGggOC40IEMgLTAuNTUzLDkuNjM2IDAsOS4wODMgMCw4LjQgWiIgLz48L2c+PGcKICAgICAgICAgICAgIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM1LjY3MzgsNjguODQzOCkiCiAgICAgICAgICAgICBpZD0iZzc5NiI+PHBhdGgKICAgICAgICAgICAgICAgaWQ9InBhdGg3OTgiCiAgICAgICAgICAgICAgIHN0eWxlPSJmaWxsOiMyYTQxOWU7ZmlsbC1vcGFjaXR5OjE7ZmlsbC1ydWxlOm5vbnplcm87c3Ryb2tlOm5vbmUiCiAgICAgICAgICAgICAgIGQ9Im0gMCwwIGMgMCwtMC42ODIgLTAuNTU0LC0xLjIzNSAtMS4yMzUsLTEuMjM1IGggLTcuMTEgYyAtMC42ODIsMCAtMS4yMzYsMC41NTMgLTEuMjM2LDEuMjM1IHYgNy4xMSBjIDAsMC42ODIgMC41NTQsMS4yMzUgMS4yMzYsMS4yMzUgaCA3LjExIEMgLTAuNTU0LDguMzQ1IDAsNy43OTIgMCw3LjExIFoiIC8+PC9nPjxnCiAgICAgICAgICAgICB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyMy4zMTkzLDc1LjA2MzUpIgogICAgICAgICAgICAgaWQ9Imc4MDAiPjxwYXRoCiAgICAgICAgICAgICAgIGlkPSJwYXRoODAyIgogICAgICAgICAgICAgICBzdHlsZT0iZmlsbDojZWQxZTI0O2ZpbGwtb3BhY2l0eToxO2ZpbGwtcnVsZTpub256ZXJvO3N0cm9rZTpub25lIgogICAgICAgICAgICAgICBkPSJtIDAsMCBjIDAsLTAuNjgzIC0wLjU1MywtMS4yMzUgLTEuMjM0LC0xLjIzNSBoIC00LjkyNCBjIC0wLjY4MiwwIC0xLjIzNiwwLjU1MiAtMS4yMzYsMS4yMzUgdiA0LjkyMyBjIDAsMC42ODIgMC41NTQsMS4yMzYgMS4yMzYsMS4yMzYgaCA0LjkyNCBDIC0wLjU1Myw2LjE1OSAwLDUuNjA1IDAsNC45MjMgWiIgLz48L2c+PGcKICAgICAgICAgICAgIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjk2ODgsNjkuMTQ3OSkiCiAgICAgICAgICAgICBpZD0iZzgwNCI+PHBhdGgKICAgICAgICAgICAgICAgaWQ9InBhdGg4MDYiCiAgICAgICAgICAgICAgIHN0eWxlPSJmaWxsOiNmNzkwMTE7ZmlsbC1vcGFjaXR5OjE7ZmlsbC1ydWxlOm5vbnplcm87c3Ryb2tlOm5vbmUiCiAgICAgICAgICAgICAgIGQ9Ik0gMCwwIEMgMCwtMC42ODIgLTAuNTU0LC0xLjIzNSAtMS4yMzYsLTEuMjM1IEggLTQuNjggYyAtMC42ODIsMCAtMS4yMzYsMC41NTMgLTEuMjM2LDEuMjM1IHYgMy40NDQgYyAwLDAuNjgzIDAuNTU0LDEuMjM2IDEuMjM2LDEuMjM2IGggMy40NDQgQyAtMC41NTQsNC42OCAwLDQuMTI3IDAsMy40NDQgWiIgLz48L2c+PGcKICAgICAgICAgICAgIHRyYW5zZm9ybT0idHJhbnNsYXRlKDYwLjIwMDIsNzIuMzgyOCkiCiAgICAgICAgICAgICBpZD0iZzgwOCI+PHBhdGgKICAgICAgICAgICAgICAgaWQ9InBhdGg4MTAiCiAgICAgICAgICAgICAgIHN0eWxlPSJmaWxsOiNmOWY5Zjk7ZmlsbC1vcGFjaXR5OjE7ZmlsbC1ydWxlOm5vbnplcm87c3Ryb2tlOm5vbmUiCiAgICAgICAgICAgICAgIGQ9Im0gMCwwIGMgMCwtMC42ODIgLTAuNTU0LC0xLjIzNSAtMS4yMzUsLTEuMjM1IGggLTEuNDEyIGMgLTAuNjgyLDAgLTEuMjM2LDAuNTUzIC0xLjIzNiwxLjIzNSB2IDEuNDEyIGMgMCwwLjY4MiAwLjU1NCwxLjIzNCAxLjIzNiwxLjIzNCBoIDEuNDEyIEMgLTAuNTU0LDIuNjQ2IDAsMi4wOTQgMCwxLjQxMiBaIG0gLTEuNDc5LC0xMC4yNiBjIDAsLTAuNjgyIC0wLjU1MywtMS4yMzUgLTEuMjM2LC0xLjIzNSBoIC00LjU1MyBjIC0wLjY4MiwwIC0xLjIzNiwwLjU1MyAtMS4yMzYsMS4yMzUgdiA0LjU1NCBjIDAsMC42ODMgMC41NTQsMS4yMzYgMS4yMzYsMS4yMzYgaCA0LjU1MyBjIDAuNjgzLDAgMS4yMzYsLTAuNTUzIDEuMjM2LC0xLjIzNiB6IG0gLTguNTk2LC0xMC42MjkgYyAwLC0wLjY4MSAtMC41NTQsLTEuMjM1IC0xLjIzNiwtMS4yMzUgaCAtOC4xNTkgYyAtMC42ODEsMCAtMS4yMzQsMC41NTQgLTEuMjM0LDEuMjM1IHYgOC4xNTkgYyAwLDAuNjgyIDAuNTUzLDEuMjM1IDEuMjM0LDEuMjM1IGggOC4xNTkgYyAwLjY4MiwwIDEuMjM2LC0wLjU1MyAxLjIzNiwtMS4yMzUgeiBtIC0yLjAzMywyNy43NTMgYyAwLDAuNjgzIDAuNTUyLDEuMjM2IDEuMjM0LDEuMjM2IGggMi45ODMgYyAwLjY4MiwwIDEuMjM2LC0wLjU1MyAxLjIzNiwtMS4yMzYgViAzLjg4MiBjIDAsLTAuNjgyIC0wLjU1NCwtMS4yMzYgLTEuMjM2LC0xLjIzNiBoIC0yLjk4MyBjIC0wLjY4MiwwIC0xLjIzNCwwLjU1NCAtMS4yMzQsMS4yMzYgeiBNIDEwLjQ0NiwtMTMuNDIzIEMgOS42NzQsLTEzLjgxIDkuNDgsLTE0LjAwMyA5LjQ4LC0xMi45NCB2IDYuMjc1IGMgMCwxLjQ1MiAwLjU0OCwyMC45NTEgLTIxLjkxNiwyMC45NTEgLTYuMDg5LDAgLTExLjY0OCwtMS41NDUgLTExLjY0OCwtMi45OTQgMCwtMS40NDggLTAuMDMzLC0zLjQ3NSAtMC4wMzMsLTQuNjM0IDAsLTEuMjQ0IDAuNjc1LC0xLjg1MiAxLjY0MSwtMS44NTIgaCA1LjYgYyAwLjY3NywwIDEuNzM3LC0wLjA3MSAxLjczNywtMS4yNjUgdiAtNi4yNDggYyAwLC0xLjY0MSAtMC44MDksLTEuODU4IC0xLjY3OCwtMS44NTggaCAtNS43OTIgYyAtMC44NjksMCAtMS43MDIsLTAuNDU5IC0xLjcwMiwtMS41MjEgdiAtNS44NzMgYyAwLC0wLjg1NCAtMC4zODUsLTEuNTYxIC0xLjY0LC0xLjU2MSBoIC01LjgxMyBjIC0wLjc3MiwwIC0xLjYwNiwtMC45NjQgLTEuNjA2LC0yLjQxMiAwLC0xLjQ0OCAtMC4wMTYsLTQuNDQzIC0wLjAxNiwtNS4zMTIgMCwtMC44NjkgLTAuNzcyLC0wLjc3MiAtMi43MDMsLTAuNzcyIDAsMCAwLC0yLjc5OSAxLjI1NSwtNS45ODYgMC42NzEsLTEuNzA0IDEuNzM4LC0yLjEyMyAzLjU3MiwtMS41NDQgMCwwIDAuMjIyLC00LjYxOSAxLjI4NCwtNy41MTUgMS4wNjIsLTIuODk2IDEuOTk5LC00Ljc0OCA0Ljk5MiwtNi44NzEgMi45OTQsLTIuMTI1IDMuMjgzLC0yLjcwNCA1LjMxLC00LjM0NSAyLjAyNywtMS42NDIgMi44OTYsLTMuMTg2IDguMDE0LC0zLjE4NiA1LjIxMywwIDYuNDY4LDIuMDI3IDguNTkyLDQuMTUxIDIuMTI0LDIuMTI1IDQuNTYsMi4yMDcgNy41Myw2LjY2MyAyLjUxMSwzLjc2NSAyLjY0Nyw2LjgxMiAyLjk5MywxMS4yIDAsMCAyLjcwNCwtMS4zNTIgMy41NzMsMS4zNTEgMC41MzIsMS42NTcgMi4xMjMsOS4wNzUgMi43MDMsMTMuMDMzIDAuNTc5LDMuOTU5IC0yLjUxLDIuMDI4IC0zLjI4MywxLjY0MiIgLz48L2c+PHJlY3QKICAgICAgICAgICAgIHN0eWxlPSJmaWxsOiMwMDAwMDA7ZmlsbC1vcGFjaXR5OjA7c3Ryb2tlOiNmZmZmZmY7c3Ryb2tlLXdpZHRoOjA7c3Ryb2tlLWRhc2hhcnJheTpub25lIgogICAgICAgICAgICAgaWQ9InJlY3Q2MDEiCiAgICAgICAgICAgICB3aWR0aD0iOS45OTgxODIzIgogICAgICAgICAgICAgaGVpZ2h0PSI4LjMwNjE4MTkiCiAgICAgICAgICAgICB4PSI0NS43NzU3MTEiCiAgICAgICAgICAgICB5PSItODEuODYyODAxIgogICAgICAgICAgICAgdHJhbnNmb3JtPSJzY2FsZSgxLC0xKSIgLz48L2c+PC9nPjwvZz48L2c+PC9zdmc+Cg==)]()

</div>

* [News](#news)
* [Overview](#overview)
* [Getting Started](#getting-started)
* [Indexing & QAing on your Custom Corpora](#indexing--qaing-on-your-custom-corpora)
* [Advanced Settings](#advanced-settings)
* [Layout](#layout)
* [TODOs](#todos)
* [Acknowledgements](#acknowledgements)
* [Citation](#citation)

---

## News

- May 1, 2026: Our paper is accepted in **ICML'26**! See you in Seoul! 🎉
- May 1, 2026: Our $\Psi$-RAG is out! 🎉 

## Overview

$\Psi$-RAG is an efficient and powerful hierarchical tree-based RAG framework designed to tackle complex information-seeking scenarios. It features a *hierarchical abstract tree index* with different abstraction strategies, enabling efficient and precise retrieval with logarithmic time. It employs a *multi-granular agentic retriever* including a powerful Reading & Answering agent with a hybrid retrieval pipeline for diverse user requests. 

![overview](fig/overview.png)

### ✨Key Features

- **🌳 Corpus-Level Tree Index**: Generalizes Tree-RAG from passage-level saplings to corpus-level large trees with millions of tokens. Abstration instead of named entity recognition: organize your corpus like a library with <3 hours indexing / >1 million tokens on two 48G RTX4090 GPUs!
- **🎯 Distribution-Adaptive Indexing**: No need for explicit document structure. No need for searching for the optimal cluster numbers. No need for handpicking the initial point. No need for dimension reduction steps. No need for worrying your imbalanced data. A hierarchical tree knows it all! 
- **⚡ Efficient Indexing Techniques**: Extra bucketing and HNSW support for sonic-speed similarity ranking on corpora with 10M+ tokens!
- **📚 Flexible Abstraction Mechanism**: Choose one that you prefer: summaries💊 or keywords💊! 
- **👨‍👩‍👧‍👦 Multi-granular and Multifunctional Retrieval Pipeline**: Iterative agentic retrieval empowers cross-document multi-hop tree search. Hybrid retrieval with BM25 navigates fine-grained tree search. Natural reranker support to maximize structured RAG performance. 
- **🧑‍💻 Custom Framework Support**: Built entirely with open-source LLMs. Change backbone models at will like changing ornaments for your Christmas tree!


## Getting Started

### Package requirements

See `requirements.txt` for <img src="https://cdn.simpleicons.org/python/000/fff" alt="Node" align=center width=16> package versions. 
Install dependencies:

```bash
pip install -r requirements.txt
```

Optional packages:

- Add `pip install vllm` if you prefer vLLM.
- Add `pip isntall transformers==4.46.0` for the embedding model "nvidia/NV-Embed-V2" (indexing only).
- Add `pip install mineru[all]` if you want to build custom indexes with your local PDF files.
- Add `pip install py7zr` or `pip install rarfile` if you want to upload a .7z or .rar package of local PDF files.

### Set environment variables and run the local LLM server

You may first set some useful environment variables before running our code, including:

```bash
# Hugging Face / model download
export CUDA_VISIBLE_DEVICES=0,1
export HF_TOKEN=<your_hugging_face_token>
export HF_ENDPOINT=https://hf-mirror.com	# Hugging Face mirror for users from China
# OpenAI-compatible API backends
export OPENAI_BASE_URL=<your_base_url>
export OPENAI_API_KEY=<your_api_key>
```

For <img src="https://cdn.simpleicons.org/ollama/000/fff" alt="Node" align=center width=16>Ollama users:

```bash
export OLLAMA_NUM_PARALLEL=16
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_NUM_GPU=2
export OLLAMA_FLASH_ATTENTION=1
# run ollama
ollama serve
# pull models you want to use
ollama pull qwen3-embedding:8b
ollama pull llama3.3:latest
```

### Configuration

We provide benchmark datasets in [`data/`](data/). Configs are Python files under [`conf/`](conf/). Each file defines a `conf` dict that overrides the defaults in [`conf/__init__.py`](conf/__init__.py). See [`conf/__init__.py`](conf/__init__.py) for detailed settings and explanations.

<details>
    <summary>Important fields</summary>

```python
# ======================================= Data config =====================================
dataset: str  					# Dataset name.
data_dir: Path  				# Dataset directory, default to "data/".
test_samples: int               # Number of samples for testing with part of the dataset. 
# =================================== Embedding config ===================================
embed_name: str					# Model name for embedding documents and queries, "[PLATFORM]:[MODEL_NAME]"
embed_cache_dir: Path			# Cache directory of the embedding model, default to os.environ["HF_HOME"]
embed_model_kwargs: Dict        # Model kwargs for embedding model. Keys depend on the model you use.
# ================================= Tree indexing config =================================
tree_builder: str               # Tree builder backend. "hnsw" for HNSW-based approximate similarity ranking.
passage_as_tree: bool			# Whether to activate single-document retrieval setup.
bucket_size: int                # Maximum bucket size for bucketing. Set to None to disable it.
# =================================== Abstraction config ==================================
abs_name: str					# Model name for node abstraction, "[PLATFORM]:[MODEL_NAME_OR_PATH]"
abs_cache_dir: Path				# Cache directory of the abstraction agent, default to os.environ["HF_HOME"].
abs_model_kwargs: Dict          # Model kwargs for the abstraction agent, similar to embed_model_kwargs.
abstract_type: str				# Type of abstract. "summary" for summative text and "keyword" for keywords. 
force_index_from_scratch: bool	# Recreate embeddings and trees even if saved files exist in save_dir.
# ==================================== R&A Agent config ===================================
qa_name: str 					# Model name for agentic retrieval and QA, "[PLATFORM]:[MODEL_NAME]"
qa_cache_dir: Path 				# Cache directory of the R&A agent, default to os.environ["HF_HOME"].
qa_model_kwargs: Dict           # Model kwargs for the R&A agent, similar to embed_model_kwargs.
answer_type: str 				# The expected answer type, ("short", "medium", "long").
no_retrieval: bool              # Skip retrieval entirely and directly answer the question.
max_retrieval_time: int 		# Maximum time of retrieval attempts (total attempts - 1). Default to "auto". 
multithreading_qa_batch_size: int		# Multithread QA for efficient reproduction on large corpus.
force_qa_from_scratch: bool		# Reanswering questions even if the answer result file exists in save_dir.
# ================================ General retrieval config ===============================
tree_top_k: int 				# Number of retrieved documents from the tree retriever.
# ================================= Hybrid retrieval config ================================
hybrid_search: bool				# If sparse keyword search (BM25) is enabled.
force_sparse_index_from_scratch: bool 	# Rebuild the sparse token vocab even if saved files exist in save_dir.
sparse_top_k: int				# Number of retrieved documents by sparse keyword search. 
# ==================================== Reranking config ===================================
rerank: bool					# If reranking is enabled.
rerank_name: str				# Model name for reranking, "[PLATFORM]:[MODEL_NAME]"
rerank_cache_dir: Path 			# Cache directory of the reranking model, default to os.environ["HF_HOME"]
rerank_model_kwargs: Dict       # Model kwargs for the reranker, similar to embed_model_kwargs.
rerank_top_k: int 				# Number of final returned documents by the reranker.
# ===================================== Other config ======================================
save_dir: Path					# Save directory of everything intermediate. Set to None to skip saving.
verbose: bool					# Whether to output the detail information during QA.
```

</details>

A test example on MuSiQue with hybrid agentic search: 

```python
# conf/<custom_conf>.py
conf = {
    "dataset": "musique",
    "test_samples": 10,     # only use the first 10 documents / chunks (depending on your dataset format)
    "max_retrieval_time": 3, 
    "tree_top_k": 10, 
    "hybrid_search": True, 
    "sparse_top_k": 10, 
    "rerank_top_k": 5,      # fuse two result sets by reciprocal rank fusion and take top 5 as the augmented context
}
```

### Run the pipeline

Save your config file as `conf/<custom_conf>.py`, and:

(1) **Build the tree index**

```bash
python index.py --config <custom_conf>
```

The pickle file containing the tree index will be saved in `<save_dir>/`  ([`output/`](output/) by default). If hybrid search is enabled, a `bm25_<dataset_name>` directory containing the sparse index will also be created.

> You can also temporarily change some configurations via CLI arguments. This will override your `<custom_conf>.py`:
> 
> ```bash
> python index.py --config <custom_conf> --force_index_from_scratch --max_tokens_per_chunk 500
> ```

(2) **Retrieve and answer the question**

Enter the interactive chat mode with your LLM:

```bash
python qa.py --config <custom_conf> --chat                          # load the tree index and chat with your R&A agent
python qa.py --config <custom_conf> --chat -q "<your_query>"        # add an initial query. `--query` and `--question` work too
python qa.py --config <custom_conf> --tree_id 2 -q "<your_query>"   # specify the tree id if your index has multiple trees
```

Answers will be saved in `<save_dir>/results/<custom_conf>.json`. You can also use the single-turn QA mode by only specifying `-q`. If you ask the same question again, the answer will be immediately loaded from the JSON file instead of another LLM call. 

```bash
python qa.py --config <custom_conf> -q "<your_query>"               # Non-interactive single-turn QA mode
python qa.py --config <custom_conf> -q "<your_query>" --update      # force to re-answer a question and update the result file
```

For evaluation purposes (queries in the dataset file), run question answering and evaluate the results with

```bash
python qa.py --config <custom_conf> 
python eval.py --config <custom_conf>
```

### An end-to-end pipeline for reproduction

We have provided our result files in [`output/results/`](output/results/) and the corresponding preset configs in [`conf/`](conf/). Simply run `main.py` with the config files for our evaluation results, for example: 

```bash
python main.py --config musique_summary
```

Our experiments are conducted on Ubuntu 20.04.6 LTS with CUDA 12.8 and Python 3.13.5.

**Re-answer the questions based on the existing tree index**:
Set `"force_qa_from_scratch": True` to automatically download our tree indexes from [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-repo-yellow?logo=huggingface)](https://huggingface.co/Newiz430/Psi-RAG) to `<save_dir>/`. You can also manually download them and put them into `<save_dir>/`. When QA is finished, a result JSON file (sharing a name with your config file) will be saved in the `<save_dir>/results/`. 

**Rebuild tree indexes**:
Set `"force_index_from_scratch": True` to rebuild tree indexes from scratch. Set `"force_sparse_index_from_scratch": True` to rebuild BM25 indexes. 
You can also set `"save_dir": None` for a single run without saving. 

> ⚠Note that running indexing / sparse indexing / QA from scratch will **cover your existing save files** in `<save_dir>/`!

## Indexing & QAing on your Custom Corpora

> $\Psi$-RAG now supports users uploading local documents in PDF format. <img src="https://opendatalab.github.io/MinerU/images/MinerU-logo.png" alt="Node" align=center width=72> is used for PDF parsing. Three steps before running `index.py`:
> 
> - Install MinerU in your environment: `pip install mineru[all]`
>   - It is recommended to download the pipeline model locally to avoid hitting the usage limit of MinerU if you need to process batches of PDF files: 
>    ```bash
>    mineru-models-download
>    set MINERU_MODEL_SOURCE=local
>    ```
>    See [MinerU documentation](https://opendatalab.github.io/MinerU/) for more info.
> - Set `"data_dir"` to the path of (a) your PDF file, (b) your folder containing PDFs, or (c) your archive package of PDF files.
> - Add in your config file: (a) `"read_local_pdf": "file"`, (b) `"read_local_pdf": "dir"`, (c) `"read_local_pdf": "package"`. See [`conf/__init__.py`](conf/__init__.py) for parameter details.

### Step 1: Data preprocessing

Put your corpus file (filename: `<dataset_name>.<ext>`) in [`data/`](data/). Your corpus file could be formatted as a long string, a list of small chunks from a document, a list of documents, or a list of chunk lists from multiple documents.

> For evaluation purposes, `<dataset_name>.<ext>` should include the ground-truth answers for QA evaluation, and some form of supporting documents for retrieval evaluation. If there is also an independent corpus file, name it `<dataset_name>_corpus.<ext>` instead. You may refer to the existing dataset files, e.g., [`2wikimultihopqa.json`](data/2wikimultihopqa.json) and [`2wikimultihopqa_corpus.json`](data/2wikimultihopqa_corpus.json). If the ground-truth answers or supporting documents are missing, a ⚠warning will be raised. 

Then, make changes in [`src/dataset.py`](src/dataset.py):

- Add your dataset name in `dataset_pool`
- Add a conditional branch in `load_data()` and `preprocess()` to process document texts (and preset queries)
- If ground-truth answers are provided, add a conditional branch in `get_gold_answers()`
- If supporting documents are provided, add a conditional branch in `get_gold_docs()`

### Step 2: Customize prompts

Default prompt files in our experiments are in [`src/prompt/`](src/prompt/). You may rewrite the prompts (and in-context examples) for your own needs:
- `rag_agent_system_short`|`rag_agent_system_medium` in [`rag_agent.py`](src/prompt/rag_agent.py): QA prompt when `"answer_type": "short"`|`"medium"`
- `rag_qa_system_long` in [`rag_qa.py`](src/prompt/rag_qa.py): summarization prompt when `"answer_type": "long"`.

### Step 3: Run the scripts

Prepare a config file in [`conf/`](conf/) and run `index.py`. See [Configuration](#configuration) and [Run the Pipeline](#run-the-pipeline).

> For *single-document setting* (a tree index for each document), set `"passage_as_tree": True` in your config file.
> 
> For *cross-document setting* (a tree for the entire corpus): 
> - If your corpus is read as **a list of long documents**, set `"force_split": True` to split the text to chunks with size no more than ` max_tokens_per_chunk`. The tree indexing will **automatically merge the chunks from the same document** (i.e., connect them to one abstract node) to preserve semantic coherence within the document; if you want to re-cluster the chunks, set `"reorganize_leaf": True` at the same time.
> - If your corpus is read as **a list of chunk lists**, set `"force_split": True` will re-split them, otherwise the preset chunks will be used as leaf nodes.

### Step 4: Train a query hop discriminator for multi-hop queries

Before running `qa.py`, set `"max_retrieval_time": "auto"` to employ a 2-layer MLP as a lightweight query hop discriminator $\mathcal{Q}$. It reuses the query embedding and predicts the hops of the query to automatically decide the maximum retrieval time with very small latency.

By default, $\mathcal{Q}$ is trained on the query sets of the provided datasets: NQ, HotpotQA, 2Wiki, MuSiQue, and MultiHop-RAG. Currently only supports a maximum of 4 hops. The checkpoints are saved to `<save_dir>/hop_discriminator/` so do not set `"save_dir": None`. 

## Advanced Settings


<details>
    <summary>Change LLM backbones</summary>

### Change LLM backbones

You may change backbone LLMs with:

```python
conf = {
    "embed_name": "ollama:qwen3-embedding:0.6b",
    "abs_name": "vllm:meta-llama/Llama-3.1-8B-Instruct",
    "qa_name": "api:openai/gpt-5-mini",
    "rerank_name": "transformers:BAAI/bge-reranker-large",
    ...
}
```
</details>


<details>
    <summary>Building a large tree with Bucketing + HNSW Builder</summary>

### Building a large tree with Bucketing + HNSW Builder

Set `bucket_size` to enable **bucketing** on very large corpora:

```python
conf = {
    "bucket_size": 4096,    # This is an upper bound for recursive bucket splitting instead of a target bucket count
    "bucket_max_fanout": 32,
    "bucket_sample_size": 8192,
    "bucket_kmeans_iters": 5,
    "tree_save_chunk_size": 200000,
    ...
}
```

This first partitions a large corpus into multiple buckets using a fast recursive spherical $k$-means. Then a bucket tree is built inside each bucket. They are finally merged at the highest layer similar to the construction of each bucket tree. The entire tree will be saved as a directory (chunked node files + a manifest).

Use **approximate ranking via a Hierarchical Navigable Small World (HNSW) graph** to replace the global similarity ranking for industrial-scale corpora, supporting both original and bucketing tree-building:

```python
conf = {
    "tree_builder": "hnsw",
    "hnsw_top_k": 32,
    "hnsw_m": 16,
    "hnsw_ef_construction": 200,
    "hnsw_ef_search": 64,
    ...
}
```

Set `"tree_build_diagnostics": True` to print dataset token statistics, execution time of each step, memory estimates, and tree connectivity checks to make sure it works as intended.

</details>

<details>
    <summary>A quick peek inside your tree</summary>

### A quick peek inside your tree

Run 🐦‍⬛[`woodpecker.py`](woodpecker.py) for a peek at the detailed stats of a tree index file:

~~~bash
python woodpecker.py --config <custom_conf>               # Overall stats of the tree index file.
python woodpecker.py --config <custom_conf> --tree_id 1   # Stats of a specific tree (single-document setting). Never look up the wrong tree.
python woodpecker.py --config <custom_conf> --layer 5     # Stats of a specific layer (0: root) of a single tree. `--layer root` or `--layer leaf` are also valid.
python woodpecker.py --config <custom_conf> --id 10000    # Stats of a specific node from a single tree. Set `--text_only` to only print its text.
~~~

Detailed stats include:

- Tree: dataset name, #tokens, #trees, tree path, tree size (GB), tree builder, connectivity, #layers, #nodes per layer, #tokens per node, embedding dimension, #children stats 
- Layer: tree id, current layer, #nodes, list of node ids
- Node: id, layer, text, ancestor ids, brief ancestor texts, children ids, brief children texts

</details>

## Layout

<details>
    <summary>Layout</summary>

```
Psi-RAG/
|_ README.md
|_ requirements.txt # Dependency list.
|_ index.py         # Builds the tree index and sparse index.
|_ qa.py            # Loads an existing tree index, retrieves context, and answers questions.
|_ eval.py          # Evaluates saved QA results.
|_ main.py          # End-to-end entrypoint: runs indexing, QA, and evaluation in one script.
|_ woodpecker.py    # Prints statistics of your tree index.
|
|_ conf/                    # Configurations.
|  |_ __init__.py           # Introduces configs and sets default values.
|  |_ <dataset>_summary.py  # Preset config files using summative abstracts.
|  |_ <dataset>_keyword.py  # Preset config files using keyword abstracts.
|
|_ data/                        # Benchmark datasets.
|  |_ <dataset>_corpus.<ext>    # Corpus files for indexing.
|  |_ <dataset>.<ext>           # Data files with ground-truth info. 
|
|_ fig/             # Figures used by README.md.
|
|_ log/                 # Log files.
|  |_ stdout.log        # General log tracing the stdout.
|  |_ <conf_name>.log   # Specific log tracing the LLM output and evaluation results for a single run.
|
|_ output/                  # Saved index files and result files.
|  |_ *.pkl                 # Tree indexes.
|  |_ bm25_<dataset>/       # Sparse BM25 indexes.
|  |_ hop_discriminator/    # Weights of the query hop discriminator.
|  |_ results/              # Saved LLM answers, retrieved chunks, and config.
|
|_ src/                     # Core implementation of $Psi$-RAG.
|  |_ __init__.py
|  |_ dataset.py            # Loads and preprocesses data.
|  |_ pdf.py                # Read and parse user-uploaded PDFs.
|  |_ rag.py                # High-level RAG controller: build/load/save tree, retrieve, and qa.
|  |_ tree_retriever.py     # Tree retrieval, hybrid retrieval, and reranking logic.
|  |_ hop_discriminator.py  # Implementation of the query hop discriminator.
|  |_ evaluation.py         # Metric implementations for evaluation.
|  |_ utils.py              # Shared data structures and helper functions.
|
|  |_ tree_builder/         # Tree construction algorithms.
|  |  |_ __init__.py
|  |  |_ base.py            # Base tree builder: create leaf nodes and launch tree construction.
|  |  |_ abstract.py        # Standard hierarchical abstract tree builder.
|  |  |_ ann.py             # ANN-based tree builder based on an HNSW graph.
|  |  |_ bucketed.py        # Bucketed tree builder.
|  |  |_ bucketed_exact.py  # Bucketed builder using exact similarity ranking.
|  |  |_ bucketed_hnsw.py   # Bucketed builder using HNSW-based approximate ranking.
|  |  |_ chunks.py          # Chunked save/load format for bucketed tree indexes.
|
|  |_ model/            # Backend adapters for LLMs.
|  |  |_ __init__.py
|  |  |_ embed.py       # Embedding model wrappers.
|  |  |_ abstract.py    # Abstraction model wrappers.
|  |  |_ qa.py          # QA model wrappers.
|  |  |_ rerank.py      # Reranker model wrappers.
|
|  |_ prompt/           # Prompt templates.
|  |  |_ __init__.py
|  |  |_ rag_abs.py     # Prompts and in-context examples for abstraction.
|  |  |_ rag_agent.py   # Prompts and in-context examples for agentic retrieval and multi-hop QA.
|  |  |_ rag_qa.py      # Prompts for single-hop QA, narrative QA and summarization.

```
</details>


## TODOs

- [x] Full result files & tree index files
- [x] Demos / Quick guide
- [x] Custom dataset / model code examples
- [x] Approximate similarity ranking techniques
- [x] Query hop discriminator
- [x] vLLM support
- [x] Local PDF & MinerU support
- [x] Interactive QA mode
- [ ] Demo video & project page
- [ ] Tree insertion
- [ ] Code refactoring & maintenance
- [ ] ...


## Acknowledgements

This is built mainly upon [RAPTOR](https://github.com/parthsarthi03/raptor), [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), and [HypHC](https://github.com/HazyResearch/HypHC). We sincerely thank their efforts. 

$\Psi$-RAG is released under the [MIT license](LICENSE). 

**Contributions of any kind are always welcome! Feel free to add a fix, a new feature, or anything helpful to this open-source project.** 🥰

## Citation

Please consider citing our work if it helps:

```bibtex
@inproceedings{psi-rag,
  title={Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation}, 
  author={Ziwen, Zhao and Menglin, Yang},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026},
  month={July},
  address={Seoul, South Korea},
  pages={TBD},
  publisher={TBD},
}
```