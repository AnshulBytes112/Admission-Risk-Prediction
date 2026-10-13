param([string]$HdfsPath = "/emergency-admission-bda")
$ErrorActionPreference = "Stop"
Write-Host "HDFS overview:"
docker compose exec namenode hdfs dfs -ls -R $HdfsPath
if ($LASTEXITCODE -ne 0) { throw "Could not list HDFS path. Check NameNode health and destination path." }
Write-Host "`nDisk usage:"
docker compose exec namenode hdfs dfs -du -h -s $HdfsPath
