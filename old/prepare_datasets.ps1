# Define the parent directory and the original directory name
$parentDir = "C:\Users\Martin\Documents\recsys-bias-propagation-simulation\data"
$originalDirName = "eoci_dataset_itemknn"

# Define the model names
$modelNames = @("_bpr", "_neumf", "_propensitymf", "_blindspotaware", "_multivae")

# Iterate over each model name
foreach ($modelName in $modelNames) {
    # Copy the original directory
    $newDirPath = Join-Path -Path $parentDir -ChildPath ($originalDirName -replace "_itemknn", $modelName)
    Copy-Item -Path (Join-Path -Path $parentDir -ChildPath $originalDirName) -Destination $newDirPath -Recurse

    # Rename files in the new directory
    Get-ChildItem -Path $newDirPath -Recurse | 
    Where-Object { $_.Name -match "_itemknn" } | 
    Rename-Item -NewName { $_.Name -replace "_itemknn", $modelName }
}

Write-Host "Directories copied and files renamed successfully."
