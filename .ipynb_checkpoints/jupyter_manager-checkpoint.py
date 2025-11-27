import subprocess
import threading
import time
import re
import requests
from pyngrok import ngrok

class JupyterManager:
    def __init__(self):
        self.jupyter_process = None
        self.ngrok_tunnel = None
        self.jupyter_token = None
        self.ngrok_url = None
    
    def start_services(self):
        try:
            # Detener servicios previos
            self.stop_services()
            
            # Iniciar Jupyter en segundo plano
            self.jupyter_process = subprocess.Popen(
                ["jupyter", "lab", "--ip=0.0.0.0", "--port=8080", "--no-browser", "--allow-root"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Esperar a que Jupyter inicie y extraer token
            time.sleep(3)
            self._extract_jupyter_token()
            
            # Iniciar Ngrok
            self.ngrok_tunnel = ngrok.connect(8080, "http")
            self.ngrok_url = self.ngrok_tunnel.public_url
            
            return {
                "success": True,
                "jupyter_token": self.jupyter_token,
                "ngrok_url": self.ngrok_url
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_jupyter_token(self):
        """Extrae el token de la salida de Jupyter"""
        time.sleep(2)
        try:
            # Leer salida de Jupyter para encontrar token
            output, _ = self.jupyter_process.communicate(timeout=1)
            token_match = re.search(r'token=([a-f0-9]+)', output)
            if token_match:
                self.jupyter_token = token_match.group(1)
            else:
                self.jupyter_token = "No encontrado - Revisar manualmente"
        except:
            self.jupyter_token = "Ejecutando - Revisar manualmente"
    
    def stop_services(self):
        try:
            if self.jupyter_process:
                self.jupyter_process.terminate()
                self.jupyter_process = None
            
            if self.ngrok_tunnel:
                ngrok.disconnect(self.ngrok_tunnel.public_url)
                self.ngrok_tunnel = None
            
            self.jupyter_token = None
            self.ngrok_url = None
            
            return {"success": True, "message": "✅ Servicios detenidos"}
        except Exception as e:
            return {"success": False, "message": f"❌ Error: {str(e)}"}
    
    def get_status(self):
        jupyter_status = "✅ Ejecutando" if self.jupyter_process else "❌ Detenido"
        ngrok_status = "✅ Activo" if self.ngrok_url else "❌ Inactivo"
        
        status_text = (
            f"📊 Estado de Servicios:\n\n"
            f"Jupyter Lab: {jupyter_status}\n"
            f"Ngrok: {ngrok_status}\n"
        )
        
        if self.ngrok_url and self.jupyter_token:
            status_text += f"\n🔗 URL: {self.ngrok_url}\n"
            status_text += f"🔑 Token: {self.jupyter_token}\n"
            status_text += f"📎 Acceso directo: {self.ngrok_url}/lab?token={self.jupyter_token}"
        
        return status_text