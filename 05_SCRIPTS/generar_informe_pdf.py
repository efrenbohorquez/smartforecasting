import markdown
import os
import datetime

# Configuración
DIR_DOCS = r"d:\deep\entregadinal 22nov\04_DOCUMENTACION\Informes_Tecnicos"
DIR_PRESENTACION = r"d:\deep\entregadinal 22nov"
OUTPUT_FILE = os.path.join(DIR_DOCS, "INFORME_FINAL_COMPLETO.html")

# Archivos a incluir
FILES = [
    {
        "path": os.path.join(DIR_DOCS, "resumen_ejecutivo.md"),
        "title": "Resumen Ejecutivo"
    },
    {
        "path": os.path.join(DIR_DOCS, "informe_tecnico_completo.md"),
        "title": "Informe Técnico Completo"
    },
    {
        "path": os.path.join(DIR_PRESENTACION, "GUION_PRESENTACION_10MIN.md"),
        "title": "Guion de Presentación"
    }
]

# CSS para impresión PDF
CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body { 
        font-family: 'Roboto', sans-serif; 
        line-height: 1.6; 
        color: #333; 
        max-width: 800px; 
        margin: 0 auto; 
        padding: 40px;
    }
    
    @media print {
        body { max-width: 100%; padding: 20px; }
        .no-print { display: none; }
        a { text-decoration: none; color: #333; }
    }
    
    h1 { color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 10px; margin-top: 50px; page-break-after: avoid; }
    h2 { color: #283593; margin-top: 30px; page-break-after: avoid; }
    h3 { color: #3949ab; margin-top: 20px; }
    
    table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }
    th, td { border: 1px solid #e0e0e0; padding: 10px; text-align: left; }
    th { background-color: #f5f5f5; color: #1a237e; }
    tr:nth-child(even) { background-color: #fafafa; }
    
    code { background-color: #f5f5f5; padding: 2px 5px; border-radius: 3px; font-family: Consolas, monospace; color: #c62828; }
    pre { background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; border: 1px solid #e0e0e0; }
    pre code { background-color: transparent; color: #333; padding: 0; }
    
    blockquote { border-left: 4px solid #1a237e; margin: 0; padding-left: 20px; color: #555; font-style: italic; }
    
    .page-break { page-break-before: always; }
    
    .cover { text-align: center; margin-top: 150px; margin-bottom: 150px; }
    .cover h1 { border: none; font-size: 3.5em; color: #1a237e; margin-bottom: 20px; }
    .cover h2 { color: #555; font-weight: 300; font-size: 1.8em; }
    .cover-info { margin-top: 100px; font-size: 1.2em; color: #777; }
    
    .toc { background: #f5f5f5; padding: 30px; border-radius: 8px; margin: 40px 0; }
    .toc h2 { margin-top: 0; border-bottom: none; }
    .toc ul { list-style: none; padding: 0; }
    .toc li { margin: 10px 0; }
    .toc a { text-decoration: none; color: #1a237e; font-weight: 500; }
    .toc a:hover { text-decoration: underline; }
    
    .footer { text-align: center; margin-top: 50px; font-size: 0.8em; color: #999; border-top: 1px solid #eee; padding-top: 20px; }
</style>
"""

def generate_html():
    html_content = []
    
    # 1. Portada
    html_content.append(f"""
    <div class="cover">
        <h1>Sistema de Predicción de Demanda</h1>
        <h2>Informe Técnico Final y Presentación</h2>
        <div class="cover-info">
            <p><strong>Proyecto Deep Learning</strong></p>
            <p>Maestría en Inteligencia Artificial</p>
            <p>{datetime.datetime.now().strftime('%B %Y')}</p>
        </div>
    </div>
    <div class="page-break"></div>
    """)
    
    # 2. Tabla de Contenidos
    html_content.append("""
    <div class="toc">
        <h2>Tabla de Contenidos</h2>
        <ul>
            <li><a href="#section-1">1. Resumen Ejecutivo</a></li>
            <li><a href="#section-2">2. Informe Técnico Completo</a></li>
            <li><a href="#section-3">3. Guion de Presentación</a></li>
        </ul>
    </div>
    <div class="page-break"></div>
    """)
    
    # 3. Contenido
    for i, file_info in enumerate(FILES, 1):
        try:
            with open(file_info["path"], "r", encoding="utf-8") as f:
                text = f.read()
                
            # Convertir Markdown a HTML
            html_body = markdown.markdown(text, extensions=['tables', 'fenced_code'])
            
            # Añadir sección
            html_content.append(f'<div id="section-{i}">')
            html_content.append(html_body)
            html_content.append('</div>')
            
            # Salto de página entre documentos (excepto el último)
            if i < len(FILES):
                html_content.append('<div class="page-break"></div>')
                
        except Exception as e:
            print(f"Error leyendo {file_info['path']}: {e}")
            html_content.append(f"<p style='color:red'>Error cargando sección: {file_info['title']}</p>")

    # 4. Ensamblar HTML final
    full_html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Informe Final - Proyecto Deep Learning</title>
        {CSS}
    </head>
    <body>
        {''.join(html_content)}
        <div class="footer">
            Generado automáticamente el {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}
        </div>
    </body>
    </html>
    """
    
    # Guardar archivo
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"Informe generado exitosamente en: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_html()
