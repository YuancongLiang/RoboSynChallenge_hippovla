# GitHub 与 RoboSynChallenge 提交说明

整理日期：2026-10-16。源部署目录为 `/data/milu/RoboSynChallenge_ws/robosyn_starvla_policy_server`，选用测评文档指定的 `hippo-new-model/final_model/pytorch_model.pt`。原目录和运行中的服务未修改。

## 官网当前状态

[Submit Policy](https://robosyn-bench.net/#/submit-policy) 的公开前端当前设置 `POLICY_SUBMISSION_INTAKE_CLOSED = true`，提示新的 evaluation requests 已关闭。FAQ 列出的初赛版本截止时间为 2026-10-11 23:59 AoE。

已有请求如收到问题邮件，应使用最新邮件中的修正表单；官网 FAQ 说明修正会固定 GitHub commit，权重按修正时间前的历史选择。本包准备于截止日之后，能否作为修正或延期版本使用，需要以主办方对已有请求的安排为准。打包完成不代表测评请求已经提交。

## 上传 GitHub

将压缩包解压，**上传解压后的项目文件**，不要只把 ZIP 当作仓库内容。项目根目录应能直接看到 `README.md`、`policy/`、`transformer/` 和 `run_policy_server.sh`。

```bash
cd robosyn_hippovla_submission
git init -b main
git add .
git commit -m "Prepare HIPPOVLA RoboSynChallenge evaluation"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
git rev-parse HEAD
```

记录最后一条命令输出的 commit。代码可使用私有仓库，但须邀请 GitHub 用户 `robosynchallenge` 为协作者并确保对方能访问。压缩包已排除 checkpoint、基础模型、环境、日志、视频、缓存和凭据。

## 上传 Hugging Face

GitHub 代码不是完整测评材料。还需另建 Hugging Face **model** 仓库上传当前约 17.9 GB 的 policy checkpoint、同次训练的配置和统计文件：

```text
YOUR_ACCOUNT/YOUR_MODEL/
├── README.md
├── config.yaml
├── dataset_statistics.json
├── checkpoint_identity.json
└── final_model/pytorch_model.pt
```

本次已另外准备本机目录 `../robosyn_hippovla_hf_upload`，其中 checkpoint 使用硬链接，不会额外复制约 17.9 GB；不要原地改写其中权重。也可自行生成：

```bash
python scripts/prepare_hf_upload.py \
  --checkpoint /path/to/hippo-new-model/final_model/pytorch_model.pt \
  --output /path/outside/github/robosyn_hippovla_hf_upload

# 使用装有 huggingface_hub CLI 的环境；登录操作在自己的终端完成。
hf auth login
hf upload YOUR_ACCOUNT/YOUR_MODEL ../robosyn_hippovla_hf_upload . --repo-type model
```

记录上传后的不可变 revision。若限制访问，官网仅支持 gated model：保持模型仓库可见，在 Settings 开启 Access requests / Manual approval，Add access 授权个人用户 `RoboSynChallenge`，核实其出现在 accepted list。官网不支持私有 HF 仓库邀请，也不接受参赛者提供访问 token。

基础模型需另外按根 README 下载。若要完全复现当前本地基础模型，也应保存其快照 revision 或在有分发权限时提供相同快照；当前目录没有记录可验证的上游 revision，文档不虚构版本号。

## 填写表单

参照 `SUBMISSION_TEMPLATE.md`。需要自行填写的事实信息：

- GitHub URL 与固定 commit。
- HF model URL 与固定 revision；同一权重服务十个任务，十个任务输入框重复填写相同地址。
- 实际训练数据来源、真实/仿真数据的范围、版本和比例。配置中的 `robosynchallenge_all` 只是本地数据混合名称，不能据此推断完整数据来源。
- 基础模型快照版本，以及注册团队信息。

`submission.json` 用 `null` 标注尚未知的链接和 revision。更新这些字段后，再生成固定 GitHub commit；不要把空字段误当作已完成提交。

## 测评范围与已知限制

十个任务全部使用同一个当前 checkpoint，具体命令见根 README。`items_handover`、`sample_loading` 不在当前训练任务列表内；其 ratio 夹爪设置沿用测评文档，仍需实测验证。`table_rearrangement` 沿用 physical 单位，训练动作与状态的单位一致性风险尚未解决。

默认完整随机环境。每任务步数从目标 RoboSynChallenge 的 `gym_config.json` 读取，不把本地旧表中的数字当作固定官方上限。`scripts/eval.py` 的 summary 只记录子进程退出状态，不宣称所有回合完成；最终成功率、完成数及异常以官方日志为准。当前整理不包含新的十任务 × 五十回合成绩。

## 来源

- [提交页与 FAQ](https://robosyn-bench.net/#/submit-policy)
- [官网前端：表单字段、关闭状态与 FAQ](https://robosyn-bench.net/static/js/pages-app.js?v=20261014-workshop4)
- [官方策略接入文档](https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html)
- `provenance.json`：本机来源与参考 checkout commit。
- `checkpoint_identity.json`：本次权重大小与 SHA-256。
