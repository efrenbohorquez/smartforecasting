import os
import datetime
import html

# Configuración
DIR_SCRIPTS = r"d:\deep\entregadinal 22nov\05_SCRIPTS"
DIR_OUTPUT = r"d:\deep\entregadinal 22nov\04_DOCUMENTACION\Informes_Tecnicos"
OUTPUT_FILE = os.path.join(DIR_OUTPUT, "CODIGOS_FUENTE_COMPLETOS.html")

# Scripts a incluir
SCRIPTS = [
    {
        "filename": "fase_final_red_neuronal_converted.py",
        "title": "1. Script Principal (Conversión del Cuaderno)",
        "desc": "Lógica principal de entrenamiento, optimización y generación de modelos."
    },
    {
        "filename": "analisis_completo_todos_los_datos.py",
        "title": "2. Análisis Completo de Datos",
        "desc": "Script para generar estadísticas globales y análisis masivo de bodegas."
    },
    {
        "filename": "predicciones_REALES.py",
        "title": "3. Generador de Predicciones Reales",
        "desc": "Script para cargar modelos y generar predicciones con datos nuevos."
    },
    {
        "filename": "ejemplo_uso_modelo.py",
        "title": "4. Ejemplo de Uso",
        "desc": "Demo simplificada de cómo cargar un modelo y hacer una inferencia."
    },
    {
        "filename": "prediccion_simple_real.py",
        "title": "5. Predicción Simple",
        "desc": "Versión minimalista para pruebas rápidas."
    }
]

# CSS para impresión de código
CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Fira+Code&display=swap');
    
    body { 
        font-family: 'Roboto', sans-serif; 
        line-height: 1.5; 
        color: #333; 
        max-width: 900px; 
        margin: 0 auto; 
        padding: 40px;
    }
    
    @media print {
        body { max-width: 100%; padding: 20px; }
        .no-print { display: none; }
        a { text-decoration: none; color: #333; }
        pre { white-space: pre-wrap; word-wrap: break-word; }
    }
    
    h1 { color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 10px; margin-top: 50px; page-break-after: avoid; }
    h2 { color: #283593; margin-top: 40px; border-left: 5px solid #283593; padding-left: 15px; page-break-after: avoid; }
    .desc { color: #666; font-style: italic; margin-bottom: 20px; }
    
    .code-container {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 30px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    pre {
        margin: 0;
        font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
        font-size: 0.85em;
        overflow-x: auto;
    }
    
    .page-break { page-break-before: always; }
    
    .cover { text-align: center; margin-top: 200px; margin-bottom: 200px; }
    .cover h1 { border: none; font-size: 3em; color: #1a237e; margin-bottom: 20px; }
    .cover h2 { color: #555; font-weight: 300; font-size: 1.5em; border: none; padding: 0; }
    
    .toc { background: #f5f5f5; padding: 30px; border-radius: 8px; margin: 40px 0; }
    .toc ul { list-style: none; padding: 0; }
    .toc li { margin: 10px 0; }
    .toc a { text-decoration: none; color: #1a237e; font-weight: 500; }
    
    .footer { text-align: center; margin-top: 50px; font-size: 0.8em; color: #999; border-top: 1px solid #eee; padding-top: 20px; }
    
    /* Simple Syntax Highlighting */
    .kw { color: #0000ff; font-weight: bold; } /* Keyword */
    .str { color: #a31515; } /* String */
    .com { color: #008000; } /* Comment */
    .num { color: #098658; } /* Number */
</style>
"""

def highlight_code(code):
    """Simple syntax highlighter for Python"""
    lines = code.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Escape HTML first
        line = html.escape(line)
        
        # Very basic highlighting (order matters)
        # Comments
        if '#' in line:
            parts = line.split('#', 1)
            line = parts[0] + f'<span class="com">#{parts[1]}</span>'
            
        # Strings (basic)
        if '"' in line and '<span class="com">' not in line:
            # This is very fragile, but sufficient for basic printing
            pass 
            
        formatted_lines.append(line)
        
    return '\n'.join(formatted_lines)

def generate_html():
    html_content = []
    
    # 1. Portada
    html_content.append(f"""
    <div class="cover">
        <h1>Códigos Fuente del Proyecto</h1>
        <h2>Sistema de Predicción de Demanda con LSTM</h2>
        <div style="margin-top: 50px; color: #777;">
            <p>Compendio de Scripts Python</p>
            <p>{datetime.datetime.now().strftime('%B %Y')}</p>
        </div>
    </div>
    <div class="page-break"></div>
    """)
    
    # 2. Tabla de Contenidos
    html_content.append("""
    <div class="toc">
        <h2>Índice de Scripts</h2>
        <ul>
    """)
    
    for i, script in enumerate(SCRIPTS, 1):
        html_content.append(f'<li><a href="#script-{i}">{script["title"]}</a></li>')
        
    html_content.append("""
        </ul>
    </div>
    <div class="page-break"></div>
    """)
    
    # 3. Contenido de Scripts
    for i, script in enumerate(SCRIPTS, 1):
        file_path = os.path.join(DIR_SCRIPTS, script["filename"])
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()
            
            # Formatear código
            formatted_code = highlight_code(code_content)
            
            html_content.append(f"""
            <div id="script-{i}">
                <h2>{script["title"]}</h2>
                <p class="desc">{script["desc"]}</p>
                <p style="font-family: monospace; background: #eee; padding: 5px; display: inline-block;">Archivo: {script["filename"]}</p>
                
                <div class="code-container">
                    <pre><code>{formatted_code}</code></pre>
                </div>
            </div>
            """)
            
            if i < len(SCRIPTS):
                html_content.append('<div class="page-break"></div>')
                
        except Exception as e:
            print(f"Error leyendo {script['filename']}: {e}")
            html_content.append(f"<p style='color:red'>Error cargando archivo: {script['filename']}</p>")

    # 4. Ensamblar HTML final
    full_html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Códigos Fuente - Proyecto Deep Learning</title>
        {CSS}
    </head>
    <body>
        {''.join(html_content)}
        <div class="footer">
            Documento generado automáticamente el {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}
        </div>
    </body>
    </html>
    """
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"Documento de códigos generado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_html()
