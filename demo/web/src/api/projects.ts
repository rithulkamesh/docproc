import { apiClient } from './client'

export interface Project {
  id: string
  name: string
  is_default: boolean
  created_at: string | null
  updated_at: string | null
}

export interface ListProjectsResponse {
  projects: Project[]
}

export async function listProjects(): Promise<Project[]> {
  const res = await apiClient.get<ListProjectsResponse>('/projects/')
  return res.projects
}

export async function getProject(projectId: string): Promise<Project> {
  return apiClient.get<Project>(`/projects/${encodeURIComponent(projectId)}`)
}

export async function updateProject(projectId: string, body: { name?: string }): Promise<Project> {
  return apiClient.patch<Project>(`/projects/${encodeURIComponent(projectId)}`, body)
}

export async function createProject(body: { name: string }): Promise<Project> {
  return apiClient.post<Project>('/projects/', body)
}
