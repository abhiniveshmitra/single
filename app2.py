{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "searchServiceName": {
      "type": "string",
      "defaultValue": "eus-cdr-ai-search"
    },
    "searchSku": {
      "type": "string",
      "defaultValue": "standard"
    },
    "foundryWorkspaceName": {
      "type": "string",
      "defaultValue": "myfoundryws"
    },
    "location": {
      "type": "string",
      "defaultValue": "westeurope"
    }
  },
  "resources": [
    {
      "type": "Microsoft.Search/searchServices",
      "apiVersion": "2023-11-01",
      "name": "[parameters('searchServiceName')]",
      "location": "[parameters('location')]",
      "sku": {
        "name": "[parameters('searchSku')]"
      },
      "properties": {
        "partitionCount": 1,
        "replicaCount": 1,
        "hostingMode": "default",
        "publicNetworkAccess": "Disabled"
      }
    },
    {
      "type": "Microsoft.AzureAI.Foundry/workspaces",
      "apiVersion": "2024-05-01-preview",
      "name": "[parameters('foundryWorkspaceName')]",
      "location": "[parameters('location')]",
      "properties": {}
    }
  ]
}
