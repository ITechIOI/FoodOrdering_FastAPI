import requests
from fastapi import HTTPException
from app.core.config import settings

def get_menu_items():
    graphql_endpoint = settings.NESTJS_MENU_ENDPOINT
    query = """
    {
        findAllNotPaginate {
            id
            name
            description
            imageUrl
        }
    }
    """
    try:
        response = requests.post(
            graphql_endpoint,
            json={"query": query}
        )

        # 🔍 In lỗi từ GraphQL nếu có
        print("Status code:", response.status_code)
        print("GraphQL response:", response.text)

        response.raise_for_status()

        data = response.json()

        if "errors" in data:
            raise HTTPException(status_code=502, detail=data["errors"])

        return data["data"]["findAllNotPaginate"]

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Cannot connect to NestJS GraphQL: {e}")


def filter_menu_by_ids(menu_data, ids):
    id_set = set(ids)  
    return [item for item in menu_data if str(item.get("id")) in id_set]