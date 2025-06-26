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
      "sku": {
        "name": "[parameters('skuName')]"
      },
      "kind": "OpenAI",
      "properties": {
        "apiProperties": {
          "statisticsEnabled": true
        }
      }
    }
  ]
}
