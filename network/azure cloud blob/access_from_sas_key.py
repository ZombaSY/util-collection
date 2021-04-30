from azure.storage.blob import ContainerClient

sas_token = 'sv=2019-02-02&spr=https%2Chttp&se=2020-07-17T11%3A21%3A57Z&sr=c&sp=racwd&sig=E4j45tjOrsk3TyyQHbuzUxzIQ98JbScJ156a7k9wTqQ%3D'

container: ContainerClient = ContainerClient.from_container_url(
    container_url="https://digocloud.blob.core.windows.net/mycontainer",
    credential=sas_token
)

container.upload_blob()

