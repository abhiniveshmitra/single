az account set --subscription 3936386c-bb2d-43ec-b7f1-2729bd65fcc0

az deployment group create `
  --resource-group rg-abhinivesh.mitra-1469 `
  --template-file "C:\Users\e819643\Documents\azure-ai-resources.json" `
  --parameters searchServiceName=mysearchservice `
               foundryWorkspaceName=myfoundryws `
               location=westeurope
