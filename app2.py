{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "accountName": {
      "type": "string",
      "defaultValue": "eus-cdr-openai"
    },
    "location": {
      "type": "string",
      "defaultValue": "westeurope"
    },
    "skuName": {
      "type": "string",
      "defaultValue": "S0"
    }
  },
  "resources": [
    {
      "type": "Microsoft.CognitiveServices/accounts",
      "apiVersion": "2023-05-01",
      "name": "[parameters('accountName')]",
      "location": "[parameters('location')]",
      "kind": "OpenAI",
      "sku": {
        "name": "[parameters('skuName')]"
      },
      "properties": {
        "apiProperties": {
          "statisticsEnabled": true
        }
      }
    }
  ]
}
az deployment group create --resource-group rg-abhinivesh.mitra-1469 --template-file "C:\Users\t819643\Documents\openai.json"
