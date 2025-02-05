import asyncio
from metagpt.roles import (
    Architect,
    Engineer,
    ProductManager,
    ProjectManager,
    QaEngineer,
)
from metagpt.team import Team
async def startup(idea: str):
    company = Team()
    company.hire(
        [
            ProductManager(),
            Architect(),
            ProjectManager(),
            Engineer(),
            QaEngineer(),
        ]
    )
    company.invest(investment=3.0)
    company.run_project(idea=idea)    
    await company.run(n_round=5)
async def main():
    print("启动主程序")
    await startup(idea="开发一个刷题程序")
    print("主程序结束")
if __name__ == "__main__":
    asyncio.run(main())
    