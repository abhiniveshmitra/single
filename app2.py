az role assignment list --assignee $(az ad signed-in-user show --query id -o tsv) --scope $(az storage account show --name cirrusplasdevat57381sa --query id -o tsv) --output table
