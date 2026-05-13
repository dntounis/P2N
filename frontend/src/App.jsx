import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, FileJson, ArrowRight, Download, RefreshCw } from 'lucide-react';
import './index.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Cleanup preview URL
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true);
    } else if (e.type === 'dragleave') {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file) => {
    if (!file.type.match('image.*')) {
      setError("Please upload an image file (PNG, JPG, etc.)");
      return;
    }
    setFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // Connect to the local FastAPI backend
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      
      setResult(data.results);
    } catch (err) {
      console.error(err);
      setError(err.message || "An error occurred during inference.");
    } finally {
      setLoading(false);
    }
  };

  const downloadJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `p2n_extraction_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Simple syntax highlighter for JSON
  const highlightJson = (jsonObj) => {
    const str = JSON.stringify(jsonObj, null, 2);
    return str.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
      let cls = 'json-number';
      if (/^"/.test(match)) {
        if (/:$/.test(match)) {
          cls = 'json-key';
        } else {
          cls = 'json-string';
        }
      } else if (/true|false/.test(match)) {
        cls = 'json-boolean';
      } else if (/null/.test(match)) {
        cls = 'json-boolean';
      }
      return `<span class="${cls}">${match}</span>`;
    });
  };

  return (
    <>
      <div className="bg-shapes">
        <div className="shape shape-1"></div>
        <div className="shape shape-2"></div>
      </div>
      
      <div className="app-container">
        <header className="header">
          <h1>Plot to Numbers</h1>
          <p>Instantly extract structured data, axis information, and elements from any scientific plot using our Vision-Encoder-Decoder model.</p>
        </header>

        <main className="main-content">
          {/* Left Panel: Upload */}
          <div className="glass-panel">
            {!preview ? (
              <div 
                className={`upload-zone ${isDragActive ? 'active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input 
                  ref={fileInputRef}
                  type="file" 
                  className="file-input" 
                  accept="image/*"
                  onChange={handleChange}
                />
                <UploadCloud size={64} className="upload-icon" />
                <h3>Upload Plot Image</h3>
                <p style={{marginTop: '0.5rem', color: 'var(--text-muted)'}}>Drag and drop, or click to browse</p>
              </div>
            ) : (
              <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                <img src={preview} alt="Plot preview" className="preview-image" />
                <button 
                  onClick={() => { setFile(null); setPreview(null); setResult(null); }}
                  style={{marginTop: '1rem', background: 'transparent', color: 'var(--text-muted)', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem'}}
                >
                  <RefreshCw size={16} /> Choose different image
                </button>
              </div>
            )}

            {error && (
              <div style={{marginTop: '1rem', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.3)'}}>
                {error}
              </div>
            )}

            <button 
              className="submit-btn" 
              onClick={handleSubmit}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <span className="loader"></span> Processing...
                </>
              ) : (
                <>
                  Extract Data <ArrowRight size={20} />
                </>
              )}
            </button>
          </div>

          {/* Right Panel: Results */}
          <div className="glass-panel" style={{display: 'flex', flexDirection: 'column'}}>
            <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem'}}>
              <FileJson className="upload-icon" size={24} style={{margin: 0}} />
              <h2 style={{fontSize: '1.5rem', fontWeight: 600}}>Structured Output</h2>
            </div>
            
            {result ? (
              <div className="json-viewer">
                <button className="download-btn" onClick={downloadJson}>
                  <Download size={16} /> Download JSON
                </button>
                <pre dangerouslySetInnerHTML={{ __html: highlightJson(result) }} />
              </div>
            ) : (
              <div style={{flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', textAlign: 'center', padding: '2rem'}}>
                <FileJson size={48} style={{opacity: 0.2, marginBottom: '1rem'}} />
                <p>Upload a plot and click "Extract Data" to see the generated JSON.</p>
              </div>
            )}
          </div>
        </main>
      </div>
    </>
  );
}

export default App;
