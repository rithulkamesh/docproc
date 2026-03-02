import useSWR from 'swr'
import {
  getSubmission,
  submissionStatusFromApi,
  type Submission,
  type SubmissionStatus,
  POLL_INTERVAL_MS,
} from '../api/assessments'

interface UseSubmissionPollOptions {
  assessmentId: string | null
  submissionId: string | null
  enabled: boolean
}

interface UseSubmissionPollResult {
  submission: Submission | undefined
  status: SubmissionStatus
  isPolling: boolean
  error: Error | undefined
  mutate: () => void
}

export function useSubmissionPoll({
  assessmentId,
  submissionId,
  enabled,
}: UseSubmissionPollOptions): UseSubmissionPollResult {
  const key =
    enabled && assessmentId && submissionId
      ? ['submission', assessmentId, submissionId]
      : null

  const { data: submission, error, mutate } = useSWR<Submission>(
    key,
    () => {
      if (!assessmentId || !submissionId) return Promise.reject(new Error('Missing ids'))
      return getSubmission(assessmentId, submissionId)
    },
    {
      refreshInterval: (data) => {
        if (!data) return POLL_INTERVAL_MS
        const status = submissionStatusFromApi(data.ai_status)
        if (status === 'completed' || status === 'failed') return 0
        return POLL_INTERVAL_MS
      },
      dedupingInterval: 1000,
      revalidateOnFocus: true,
    }
  )

  const status = submission ? submissionStatusFromApi(submission.ai_status) : null
  const isPolling =
    enabled &&
    !!submissionId &&
    status !== 'completed' &&
    status !== 'failed' &&
    status !== null

  return {
    submission,
    status: status ?? null,
    isPolling: !!isPolling,
    error,
    mutate,
  }
}
