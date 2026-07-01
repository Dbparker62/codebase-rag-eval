import sys, asyncio
sys.path.insert(0, "src")          # so `from tools import ...` resolves
from tools import LocalRepo

REPO_ROOT = r"C:\Users\video\Desktop\ai Vodeps"   

async def main():
    repo = LocalRepo(root=REPO_ROOT)
    print(await repo.read("comfyGen.py", 1, 20))       
    print(await repo.grep("queue_prompt")) 

asyncio.run(main())