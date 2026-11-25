import type { Component } from 'vue'

export interface NavItem { to: string; label: string; icon?: Component; num?: number }
export interface NavGroup { heading?: string; items: NavItem[] }
