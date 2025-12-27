@echo off
echo ========================================
echo CORRIGINDO CUDA NO STREAMLIT
echo ========================================
echo.

echo 1. Verificando Python atual...
python -c "import sys; print('Python:', sys.executable); import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo.
echo 2. Limpando cache do Streamlit...
if exist "%USERPROFILE%\.streamlit" (
    echo Cache do Streamlit encontrado
)

echo.
echo 3. Instalando PyTorch com CUDA no Python atual...
python -m pip uninstall torch torchvision -y
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo 4. Verificando instalacao...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo ========================================
echo CORRECAO CONCLUIDA
echo ========================================
echo.
echo Agora execute: python -m streamlit run app.py
echo.
pause

