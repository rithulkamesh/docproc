import { useState, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useWorkspace } from '../context/WorkspaceContext'

export function ProjectHeaderBar() {
  const {
    currentProject,
    setCurrentProjectName,
    documents,
    lastIndexedLabel,
    themeMode,
    setThemeMode,
    focusMode,
  } = useWorkspace()
  const [editingName, setEditingName] = useState(false)
  const [editValue, setEditValue] = useState(currentProject?.name ?? '')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setEditValue(currentProject?.name ?? '')
  }, [currentProject?.name])

  useEffect(() => {
    if (editingName && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [editingName])

  const handleSaveName = async () => {
    setEditingName(false)
    const trimmed = editValue.trim()
    if (trimmed && trimmed !== currentProject?.name) {
      await setCurrentProjectName(trimmed)
    } else {
      setEditValue(currentProject?.name ?? '')
    }
  }

  const processingCount = documents.filter((d) => d.status === 'processing').length

  return (
    <header
      className={`header-bar ${focusMode ? 'header-bar--compact' : ''}`}
      aria-label="Project header"
    >
      <div className="header-brand">
        <Link to="/">
          <span className="header-brand-label">docproc</span>
        </Link>
        {editingName ? (
          <input
            ref={inputRef}
            type="text"
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={() => void handleSaveName()}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void handleSaveName()
              if (e.key === 'Escape') {
                setEditingName(false)
                setEditValue(currentProject?.name ?? '')
              }
            }}
            aria-label="Project name"
            className="header-project-input"
          />
        ) : (
          <button
            type="button"
            onClick={() => setEditingName(true)}
            className="header-project-btn"
          >
            {currentProject?.name ?? '—'}
          </button>
        )}
      </div>
      <div className="header-meta">
        <span className="badge">
          Docs: {documents.length}
          {processingCount > 0 && ` · ${processingCount} processing`}
        </span>
        <span className="text-xs text-muted">Last indexed: {lastIndexedLabel}</span>
        <button
          type="button"
          aria-label={themeMode === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
          className="theme-toggle"
          onClick={() => setThemeMode(themeMode === 'light' ? 'dark' : 'light')}
        >
          {themeMode === 'light' ? 'Dark' : 'Light'}
        </button>
      </div>
    </header>
  )
}
