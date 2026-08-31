# Issue tracker（工单系统）：本地 Markdown（标记文本）

本项目的规格和工单保存在 `.scratch/` 目录。

## 约定

- 每项功能使用一个目录：`.scratch/<feature-slug>/`。
- 规格文件是 `.scratch/<feature-slug>/spec.md`。
- 实施工单分别保存在 `.scratch/<feature-slug>/issues/<NN>-<slug>.md`。
- 工单编号从 `01` 开始，并按前置关系排序。
- 每个文件只保存一张工单。
- `Status（状态）` 字段记录当前分诊状态。
- 后续讨论追加到文件末尾的 `Comments（讨论）` 小节。

当技能要求“发布到工单系统”时，直接在对应功能目录中新建或更新文件。
