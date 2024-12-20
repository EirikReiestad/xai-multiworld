alias get_quota='lfs quota -u $(whoami) /cluster'
alias getacc='sacctmgr show assoc where user=$(whoami) format=account --parsable2 --noheader'
alias getdefaultacc='sacctmgr show user $(whoami) format=defaultaccount --parsable2 --noheader'
alias gowork='cd /cluster/work/$(whoami)/'
alias past_jobs='sacct -X --format=JobID,Jobname%30,state,time,elapsed,nnodes,ncpus,nodelist,AllocTRES%50'
alias queue='watch -n 5 "squeue --me --format=\"%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %.19S %R\""'

alias run_job='ssh scripts/idun/run_job.sh'
