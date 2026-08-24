import argparse
from pathlib import Path

from body.agent import action_for_analysis, analyze_job
from body.models import Job, JobTarget


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="本地职位匹配分析器")
    parser.add_argument("--job", required=True, help="职位文本文件路径")
    parser.add_argument("--resume", required=True, help="简历文本文件路径")
    parser.add_argument("--title", required=True, help="职位名称")
    parser.add_argument("--company", required=True, help="公司名称")
    parser.add_argument("--city", default=None, help="职位城市")
    args = parser.parse_args()

    job = Job(
        job_id=f"local-{args.company}-{args.title}",
        title=args.title,
        company=args.company,
        city=args.city,
        description=read_text(args.job),
    )
    analysis = analyze_job(job, JobTarget(), read_text(args.resume))
    action = action_for_analysis(job, analysis)

    print(analysis.model_dump_json(indent=2, ensure_ascii=False))
    print(f"下一步动作: {action.action.value}")
    print("不会自动登录、打开平台或发送消息，请人工复制话术完成投递。")


if __name__ == "__main__":
    main()
