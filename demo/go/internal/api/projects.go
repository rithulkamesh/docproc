package api

import (
	"net/http"
	"strings"
)

// Projects: GET/POST /projects, GET/PATCH/DELETE /projects/:id
func (h *Handler) projects(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/projects")
	path = strings.Trim(path, "/")

	if path == "" || path == "/" {
		switch r.Method {
		case http.MethodGet:
			h.listProjects(w, r)
			return
		case http.MethodPost:
			h.createProject(w, r)
			return
		}
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// path = project_id
	switch r.Method {
	case http.MethodGet:
		h.getProject(w, r, path)
		return
	case http.MethodPatch:
		h.updateProject(w, r, path)
		return
	case http.MethodDelete:
		h.deleteProject(w, r, path)
		return
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
}

func (h *Handler) listProjects(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	list, err := h.pool.ListProjects(ctx)
	if err != nil {
		http.Error(w, "database error", http.StatusInternalServerError)
		return
	}
	out := make([]any, len(list))
	for i, row := range list {
		out[i] = map[string]any{"id": row.ID, "name": row.Name, "is_default": row.IsDefault}
	}
	writeJSON(w, out)
}

func (h *Handler) createProject(w http.ResponseWriter, r *http.Request) {
	var body struct{ Name string }
	if !parseBody(w, r, &body) || body.Name == "" {
		return
	}
	ctx := r.Context()
	id, err := h.pool.CreateProject(ctx, body.Name)
	if err != nil {
		http.Error(w, "database error", http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]any{"id": id, "name": body.Name, "is_default": false})
}

func (h *Handler) getProject(w http.ResponseWriter, r *http.Request, projectID string) {
	ctx := r.Context()
	row, err := h.pool.GetProject(ctx, projectID)
	if err != nil {
		http.Error(w, "database error", http.StatusInternalServerError)
		return
	}
	if row == nil {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, map[string]any{"detail": "Project not found"})
		return
	}
	writeJSON(w, map[string]any{"id": row.ID, "name": row.Name, "is_default": row.IsDefault})
}

func (h *Handler) updateProject(w http.ResponseWriter, r *http.Request, projectID string) {
	var body struct{ Name string }
	if !parseBody(w, r, &body) || body.Name == "" {
		return
	}
	ctx := r.Context()
	if err := h.pool.UpdateProject(ctx, projectID, body.Name); err != nil {
		http.Error(w, "database error", http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]any{"id": projectID, "name": body.Name, "is_default": false})
}

func (h *Handler) deleteProject(w http.ResponseWriter, r *http.Request, projectID string) {
	ctx := r.Context()
	if err := h.pool.DeleteProject(ctx, projectID); err != nil {
		http.Error(w, "database error", http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
