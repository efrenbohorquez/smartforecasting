import os
import sys

# Add project root to path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ui import create_demo

# Configuración del puerto para Render (o local)
PORT = int(os.environ.get("PORT", 7860))

if __name__ == "__main__":
    demo = create_demo()
    # server_name="0.0.0.0" es crucial para que Render exponga la app
    demo.launch(server_name="0.0.0.0", server_port=PORT)
