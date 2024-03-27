$numberOfExecutions = 20
$model = Read-Host "Enter Model Name"
$model_lower = $model.ToLower()

$dataset = Read-Host "Enter Dataset Name"
$dataset_lower = $dataset.ToLower()
$dataset_upper = $dataset.ToUpper()

$folderPath = "data\"
$pattern = "${dataset_lower}_${model_lower}_"
$commandBase = "C:\Users\Martin\AppData\Local\Programs\Python\Python39\python.exe run_loop.py ${dataset_lower}_${model_lower}"


$experimentNames = @(
    "controlGroupDE_acceptTopRandom",
    "controlGroupDE_onlyFromUS",
    "controlGroupDE_notFromUS",
    "controlGroupDE_onlyFromDE"
)

$useFPC = Read-Host "Do you want to use False Positive Correction? (yes/no)"
$fpcFlag = ""
$experimentFlag = ""
if ($useFPC -eq "yes") {
    $fpcFlag = " --use-fpc"
    $experimentFlag = "withFPC"
}

Remove-Item -Path "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\saved\$model*" -Force
Remove-Item -Path "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\saved\${dataset_lower}_$model_lower*" -Force


$flags = @(
    "--control_group --control-group-country DE --at-most-one --conditional -m ${model} --resume --top_number 10 --one-loop$fpcFlag",
    "--control_group --control-group-country DE --at-most-one --conditional -m ${model} --resume --top_number 10 --one-loop --only-from-country --country US$fpcFlag",
    "--control_group --control-group-country DE --at-most-one --conditional -m ${model} --resume --top_number 10 --one-loop --not-from-country --country US$fpcFlag",
    "--control_group --control-group-country DE --at-most-one --conditional -m ${model} --resume --top_number 10 --one-loop --only-from-country --country DE$fpcFlag"
)

for ($j = 0; $j -lt $experimentNames.Length; $j++) {
    $currentDate = Get-Date -Format "dd-MM-yy"
    $current_experiment_name = $experimentNames[$j]
    $experiment_name = "Martin_${currentDate}_${dataset_upper}_${current_experiment_name}${experimentFlag}_atMostOne"
    
    $filename="${model}_$experiment_name"
    Write-Host "Running experiment: $experiment_name saved to $filename"
    $currentFlag = $flags[$j]

    $batchContent = @"
SET tensorboard_logdir=tensorboard_$model_lower
$commandBase $currentFlag
"@

    Set-Content -Path "run_train_$model.bat" -Value $batchContent

    # Rest of the loop continue

    # Get the maximum number from existing folders
    $folders = Get-ChildItem -Path $folderPath | Where-Object { $_.PSIsContainer -and $_.Name.StartsWith($pattern) }
    $maxNumber = if ($folders.Count -eq 0) { 0 } else { 
        $folders | ForEach-Object { [int]($_.Name -replace [regex]::Escape($pattern), '') } | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum 
    }
    Write-Host "Start point found: $maxNumber"

    # Loop and execute the command
    for ($i = $maxNumber; $i -le $numberOfExecutions; $i++) {
        $outLog = "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\stdout\$filename-out$i.log"
        $errorLog = "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\stdout\$filename-error$i.log"

        Write-Host "Executing command: $command (Iteration $i)"

        Start-Process -FilePath "run_train_$model.bat" -RedirectStandardOutput "$outLog" -RedirectStandardError "$errorLog" -Wait

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Command failed in iteration $i. Exiting loop."
            break
        }
    }

    # Compress and clean up
    Compress-Archive -Path "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\tensorboard_$model\*" -Force -DestinationPath "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\log_tensorboard\$filename.zip"
    Remove-Item -Path "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\tensorboard_$model\*" -Force -Recurse

    Compress-Archive -Path "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\stdout\$filename*" -Force -DestinationPath "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\stdout_results\$filename.zip"
    Remove-Item -Path "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\stdout\$filename*" -Force

    Compress-Archive -Path "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\data\${dataset_lower}_${model_lower}*" -Force -DestinationPath "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\data\$filename.zip"
    Remove-Item -Path "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\data\${dataset_lower}_${model_lower}_*" -Force -Recurse

    Remove-Item -Path "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\saved\$model*" -Force
    Remove-Item -Path "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\saved\${dataset_lower}_$model_lower*" -Force
}

Write-Host "All experiments completed for ${model} and ${dataset_lower}."
