# Agent Skills Template Library

Biblioteca de skills para agentes CLI (OpenCode, Claude Code, etc.) basada en el estándar `.agents/skills/`.

## Instalación Rápida

```bash
# Copiar skills a tu proyecto
cp -r .agents /tu/proyecto/

# O a configuración global
cp -r .agents ~/.claude/
```

## Skills Disponibles (10)

| Skill | Descripción | Docs |
|-------|-------------|------|
| `user-setup` | Configuración + Code Review obligatorio | [SKILL.md](.agents/skills/user-setup/SKILL.md) |
| `code-review` | Revisión + Security Report estilo pentest | [SKILL.md](.agents/skills/code-review/SKILL.md) |
| `readme-makefile` | README + Makefile + Excalidraw PNGs | [SKILL.md](.agents/skills/readme-makefile/SKILL.md) |
| `code-quality` | Pylint ≥9.0 + Tests ≥85% + Reportes | [SKILL.md](.agents/skills/code-quality/SKILL.md) |
| `documentation` | Sphinx + LaTeX PDF + Whitepaper | [SKILL.md](.agents/skills/documentation/SKILL.md) |
| `git-mapper` | Commits granulares por archivo | [SKILL.md](.agents/skills/git-mapper/SKILL.md) |
| `doc-to-markdown` | DOCX/PDF/Excel → Markdown | [SKILL.md](.agents/skills/doc-to-markdown/SKILL.md) |
| `kind-cluster` | Kind cluster con GPUs (3 GPU + 2 CPU) | [SKILL.md](.agents/skills/kind-cluster/SKILL.md) |
| `file-date-mapper` | Mapeo de archivos por fechas | [SKILL.md](.agents/skills/file-date-mapper/SKILL.md) |
| `excalidraw-diagram` | Diagramas Excalidraw + PNG export | [SKILL.md](.agents/skills/excalidraw-diagram/SKILL.md) |

## Estructura

```
.agents/
├── skills/
│   ├── user-setup/
│   │   └── SKILL.md
│   ├── code-review/
│   │   ├── SKILL.md
│   │   └── references/
│   ├── readme-makefile/
│   │   ├── SKILL.md
│   │   ├── scripts/
│   │   └── references/
│   ├── code-quality/
│   │   ├── SKILL.md
│   │   ├── scripts/
│   │   └── references/
│   ├── documentation/
│   │   ├── SKILL.md
│   │   └── references/
│   ├── git-mapper/
│   │   ├── SKILL.md
│   │   └── scripts/
│   ├── doc-to-markdown/
│   │   └── SKILL.md
│   ├── kind-cluster/
│   │   └── SKILL.md
│   ├── file-date-mapper/
│   │   ├── SKILL.md
│   │   └── scripts/
│   └── excalidraw-diagram/
│       ├── SKILL.md
│       └── references/
└── skills-manifest.json
```

## Configuración

### OpenCode

Copiar `opencode.json` a la raíz del proyecto para configurar agents:

```bash
cp opencode.json /tu/proyecto/
```

### Prompts Originales

Los templates originales están en `/prompts/`:

```
prompts/
├── 0. [docker] agent_setup.md
├── 1. [docker] readme create.md
├── 2. [docker] makefile_create.md
├── 3. [docker] sphinx_create.md
├── 4. [host] docker security.md
├── 5. [docker] unit_test create.md
├── 6. [docker] python quality.md
├── 7. [docker] python quality report.md
├── 8. [docker] pdf_of_project_with_latex.md
├── 8.1. [docker] pdf_of_project_with_latex_tecnical_docs.md
├── 9. [host] git_mapper_changes.md
├── document_to_markdown_convertion.md
├── git-mapper.md
├── kind_cluster_create.md
├── mapper_files_according_datetimes.md
├── python-quality.md
├── unit-test.md
└── word_to_markdown_convertion.md
```

## Dependencias entre Skills

```
documentation ──────→ readme-makefile ──────→ excalidraw-diagram
                         ↓
                    (genera PNGs)
                         ↓
                    (usa en PDF)
```

## Autor

**wisrovi** - AI Solutions Architect

- LinkedIn: https://www.linkedin.com/in/wisrovi-rodriguez/
- GitHub: https://github.com/wisrovi/
- DockerHub: https://hub.docker.com/u/wisrovi/
- PyPI: https://pypi.org/search/?q=wisrovi/
