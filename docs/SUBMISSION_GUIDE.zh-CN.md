# GitHub 与 RoboSynChallenge 提交说明

整理日期：2026-10-16。源部署目录为 `/data/milu/RoboSynChallenge_ws/robosyn_starvla_policy_server`，选用测评文档指定的 `hippo-new-model/final_model/pytorch_model.pt`。原目录和运行中的服务未修改。

> 2026-10-24 打包修复：本次补交以主办方邮件要求为准，请使用后台追加的最新官方 response form，提交修复后的固定 commit 及全部十任务映射。具体见 [PACKAGING_CORRECTION.md](PACKAGING_CORRECTION.md)。以下包含原始提交历史说明。

## 本团队的提交窗口

主办方已批准 Luxi Tech 的个人提交例外，有效期至 2026-10-25 AoE 结束，即必须在 **2026-10-26 12:00 UTC / 北京时间 20:00 之前**完成提交。使用原先获批的注册账号和批准邮件中的现有 access token 登录，无需重新注册。公开提交关闭不影响本次账号专属批准。不要将 access token 写入代码、仓库或表单备注。提交后保存确认信息及 evaluation ID。

## 上传 GitHub

将压缩包解压，**上传解压后的项目文件**，不要只把 ZIP 当作仓库内容。项目根目录应能直接看到 `README.md`、`policy/hippovla/` 和根目录兼容入口。完整运行代码、自定义 Transformers、安装及启动脚本均位于 `policy/hippovla/`。

```bash
cd /mnt/nfs_shared/milu/robosyn_hippovla_submission
git status --short
git rev-parse HEAD
```

当前仓库为 https://github.com/YuancongLiang/RoboSynChallenge_hippovla 。将实际发布的 commit 拼接为 `https://github.com/YuancongLiang/RoboSynChallenge_hippovla/tree/<commit>`，填写到提交表单；不要填浮动的 `main` 链接。

记录最后一条命令输出的 commit。代码可使用私有仓库，但须邀请 GitHub 用户 `robosynchallenge` 为协作者并确保对方能访问。压缩包已排除 checkpoint、基础模型、环境、日志、视频、缓存和凭据。

## 上传 Hugging Face

GitHub 代码和 HF 权重都需提交。当前权重已上传至 `ovoovo/hippovla`，固定 revision 为 `352a6cf7b419734fd0b13ce75a4205c2073c03ee`，完整链接见 `SUBMISSION_TEMPLATE.md`。仓库包含 `README.md`、`config.yaml`、`dataset_statistics.json`、`checkpoint_identity.json` 和 `final_model/pytorch_model.pt`。权重约 17.9 GB，校验信息见 `checkpoint_identity.json`。

当前 HF 和 GitHub 仓库均公开。十个任务共用同一 checkpoint，表单中十个任务字段均填写相同的固定 HF revision URL。ModelScope 镜像不能替代本次要求的 HF 地址。若将来更换 checkpoint，可用 `scripts/prepare_hf_upload.py` 准备新目录。

记录上传后的不可变 revision。若限制访问，官网仅支持 gated model：保持模型仓库可见，在 Settings 开启 Access requests / Manual approval，Add access 授权个人用户 `RoboSynChallenge`，核实其出现在 accepted list。官网不支持私有 HF 仓库邀请，也不接受参赛者提供访问 token。

基础模型为 `Alibaba-DAMO-Academy/RynnBrain-8B`，固定 revision `04f588d50bf6f31b6b9f03f4d16d98dd0d22bcfe`。本地四个权重分片以及全部运行配置、tokenizer、processor 文件均与该版本的上游哈希一致，记录见 `base_model_verification.json`。按 `policy/hippovla/README.md` 下载此固定版本。

## 填写表单

参照 `SUBMISSION_TEMPLATE.md`，一次请求覆盖全部十个任务。策略训练数据来源已由团队确认：**仅使用官方 RoboSynChallenge 仿真数据，100% 仿真、0% 真实，未添加其他策略训练数据**。本地混合名称为 `robosynchallenge_all`；现有记录未包含精确数据集 release/revision，因此材料如实注明未记录。基础模型的预训练来源单独遵循 RynnBrain-8B 的说明。

`submission.json` 保存权重和基础模型的固定版本。代码 commit 取包含该文件的实际 Git commit（`git rev-parse HEAD`），避免将前一版本 commit 错当成本次修复版本。注册团队信息按主办方批准邮件填写。

## 接口和执行步数

提交版的 `policy/hippovla/` 已包含 `eval.sh`、`deploy_policy.yml` 和 `deploy_policy.py`。`eval()` 返回官方当前要求的四个值，耗时按实际预测次数记录，排除缓存动作和仿真执行。默认 `eval_action_steps: 4`、`replan_action_steps: 4`：每次预测 16 步，只执行前 4 步后从最新观测重新预测。

所有任务均使用官方 `is_task_success` 接口；提交版移除了私有成功判定、真实物体位姿读取、成功条件修改和相机修改功能。使用根 README 的安装器将新适配器安装到独立的官方评测 checkout，避免与本地旧版三返回值入口混用。

## 测评范围与已知限制

十个任务全部使用同一个当前 checkpoint，具体命令见根 README。`items_handover`、`sample_loading` 不在当前训练任务列表内；其 ratio 夹爪设置沿用测评文档，仍需实测验证。`table_rearrangement` 沿用 physical 单位，训练动作与状态的单位一致性风险尚未解决。

默认完整随机环境。每任务步数从目标 RoboSynChallenge 的 `gym_config.json` 读取，不把本地旧表中的数字当作固定官方上限。`scripts/eval.py` 的 summary 只记录子进程退出状态，不宣称所有回合完成；最终成功率、完成数及异常以官方日志为准。当前整理不包含新的十任务 × 五十回合成绩。

## 来源

- [提交页与 FAQ](https://robosyn-bench.net/#/submit-policy)
- [官网前端：表单字段、关闭状态与 FAQ](https://robosyn-bench.net/static/js/pages-app.js?v=20261014-workshop4)
- [官方策略接入文档](https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html)
- `provenance.json`：本机来源与参考 checkout commit。
- `checkpoint_identity.json`：本次权重大小与 SHA-256。
