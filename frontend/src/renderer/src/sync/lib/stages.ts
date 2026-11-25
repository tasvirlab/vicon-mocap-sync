import { Database, Zap, Film, Activity, Scan, Boxes } from '@lucide/vue'
import type { Component } from 'vue'

/** The 6 pipeline stages, in processing order. Used by Sidebar + Home so the
 *  numbering, colors, and descriptions can never drift apart. */
export interface Stage {
  num:   number
  label: string
  to:    string
  icon:  Component
  color: string   // fixed hex — stage identity color, same in both themes
  desc:  string   // one-line plain-English purpose
  needs: string   // what a new user must have ready before this step
}

export const STAGES: Stage[] = [
  {
    num: 1, label: 'C3D Convert', to: '/vicon', icon: Database, color: '#8B5CF6',
    desc: 'Turn a Vicon .c3d recording into a clean 21-joint CSV.',
    needs: 'Needs: .c3d file from Vicon Nexus',
  },
  {
    num: 2, label: 'Sync', to: '/sync', icon: Zap, color: '#10B981',
    desc: 'Align the Vicon clock with the video clock (frame offset).',
    needs: 'Needs: video + Vicon CSV + XCP file',
  },
  {
    num: 3, label: 'Overlay', to: '/overlay', icon: Film, color: '#06B6D4',
    desc: 'Draw the Vicon skeleton on the video to verify the sync.',
    needs: 'Needs: sync result + camera calibration',
  },
  {
    num: 4, label: 'REBA Score', to: '/reba', icon: Activity, color: '#F59E0B',
    desc: 'Compute ergonomic risk scores from the synced 3D joints.',
    needs: 'Needs: synced.csv from Overlay export',
  },
  {
    num: 5, label: '2D Pose', to: '/pose2d', icon: Scan, color: '#0EA5E9',
    desc: 'Run RTMPose on the clean RGB video → 2D keypoints.',
    needs: 'Needs: rgb.mp4 from Overlay export',
  },
  {
    num: 6, label: 'Dataset Export', to: '/dataset', icon: Boxes, color: '#EC4899',
    desc: 'Package everything into train/test .npz for model training.',
    needs: 'Needs: synced.csv + pose2d.csv per trial',
  },
]

export function stageFor(path: string): Stage | undefined {
  return STAGES.find(s => path.startsWith(s.to))
}
