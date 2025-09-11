docker run --rm -v ~/.ssh:/root/.ssh -v ./report:/report -w /app -e TERM=xterm wisrovi/pr_analizer sh -c "cd /scripts && sh ssh_permision.sh && python analize_complete.py --repo_owner cimacorporate --repo_name 001-AREPO "






# consola: 
# docker run --rm -v ~/.ssh:/root/.ssh -v ./report:/report -w /app -e TERM=xterm wisrovi/pr_analizer sh
