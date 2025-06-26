{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {},
  "variables": {},
  "resources": [
    {
      "type": "Microsoft.CognitiveServices/accounts",
      "apiVersion": "2023-05-01",
      "name": "eus-cdr-openai",
      "location": "westeurope",
      "kind": "OpenAI",
      "sku": {
        "name": "S0"
      },
      "properties": {
        "customSubDomainName": "eus-cdr-openai"
      }
    },
    {
      "type": "Microsoft.CognitiveServices/accounts/deployments",
      "apiVersion": "2023-05-01",
      "name": "eus-cdr-openai/gpt-4",
      "dependsOn": [
        "[resourceId('Microsoft.CognitiveServices/accounts', 'eus-cdr-openai')]"
      ],
      "properties": {
        "model": {
          "format": "OpenAI",
          "name": "gpt-4",
          "version": "1106-Preview"
        },
        "raiPolicyName": "Microsoft.Default"
      },
      "sku": {
        "name": "Standard",
        "capacity": 10
      }
    }
  ],
  "outputs": {
    "openaiEndpoint": {
      "type": "string",
      "value": "[reference(resourceId('Microsoft.CognitiveServices/accounts', 'eus-cdr-openai')).endpoint]"
    },
    "deploymentName": {
      "type": "string",
      "value": "gpt-4"
    }
  }
}
az deployment group create --resource-group abhinivesh.mitra-1469 --template-file Documents\openai.json
