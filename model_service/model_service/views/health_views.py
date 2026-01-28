from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

@api_view(['GET'])
@permission_classes([AllowAny])
def alive_check(request) -> Response:
    """
    Simple health check endpoint.
    Logic to check additional requirements can be added here. (like db connection)
    Returns:
        200 OK with {"status": "alive"} if successful.
        500 Internal Server Error with {"status": "error", "message": <error_message>} if failed.
    """
    try:
        return Response({"status": "alive"}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
