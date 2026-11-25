import { createRouter, createWebHashHistory } from 'vue-router'
import Home      from '@renderer/pages/Home.vue'
import Projects  from '@renderer/pages/Projects.vue'
import Settings  from '@renderer/pages/Settings.vue'
import Intrinsic from '@renderer/calib/pages/Intrinsic.vue'
import Extrinsic from '@renderer/calib/pages/Extrinsic.vue'
import Vicon         from '@renderer/sync/pages/Vicon.vue'
import Sync          from '@renderer/sync/pages/Sync.vue'
import Overlay       from '@renderer/sync/pages/Overlay.vue'
import Reba          from '@renderer/sync/pages/Reba.vue'
import Pose2D        from '@renderer/sync/pages/Pose2D.vue'
import DatasetExport from '@renderer/sync/pages/DatasetExport.vue'
import Analyse       from '@renderer/sync/pages/Analyse.vue'

// meta.contained = render inside the centered 1200 px column (calibration pages, Home)
export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/',          component: Home,      meta: { contained: true } },
    { path: '/intrinsic', component: Intrinsic, meta: { contained: true } },
    { path: '/extrinsic', component: Extrinsic, meta: { contained: true } },
    { path: '/vicon',     component: Vicon },
    { path: '/sync',      component: Sync },
    { path: '/overlay',   component: Overlay },
    { path: '/reba',      component: Reba },
    { path: '/pose2d',    component: Pose2D },
    { path: '/dataset',   component: DatasetExport },
    { path: '/analyse',   component: Analyse },
    { path: '/projects',  component: Projects },
    { path: '/settings',  component: Settings },
  ],
})
