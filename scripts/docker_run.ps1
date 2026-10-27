param([string]$Image = "credit-default-api", [int]$Port = 8000)
$Name = "credit-default-api"

docker rm -f $Name *> $null
docker run -d --name $Name -p "${Port}:8000" $Image | Out-Null
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

foreach ($i in 1..30) {
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:$Port/health"
        Write-Host "health: $($health.status)"
        Write-Host "API: http://127.0.0.1:$Port (docs: /docs). Stop: docker stop $Name"
        exit 0
    } catch {
        Start-Sleep -Seconds 1
    }
}

Write-Host "API did not start in 30 s. Logs:"
docker logs $Name
docker rm -f $Name *> $null
exit 1
