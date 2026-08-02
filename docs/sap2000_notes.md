# SAP2000 Notes

The full model-generation workflow requires SAP2000 on Windows with OAPI/COM access.

The dashboard and Behavior ML review workflow can be inspected without SAP2000 by loading previously generated metadata and preview files.

SAP2000 version differences can affect OAPI method signatures and result-table availability. The code checks SAP2000 return values where possible and reports API failures with contextual messages.

Large `.sdb`, `.s2k`, `.msh`, `.out`, and `.log` files are intentionally ignored by Git. Keep production-scale analysis outputs outside the repository or archive them separately.
