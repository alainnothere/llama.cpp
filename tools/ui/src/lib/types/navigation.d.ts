import type { SidebarAction } from '$lib/enums';
import type { Component } from 'svelte';

/**
 * A single clickable action in the desktop sidebar icon strip.
 */
export interface DesktopIconStripItem {
	icon: Component;
	tooltip: string;
	route?: string;
	/** URL served outside the SPA - opened in a new tab, bypassing client-side routing. */
	externalHref?: string;
	/** Custom action handled by the sidebar, e.g. opening a new-chat tab */
	action?: SidebarAction;
	activeRouteId?: string;
	activeRoutePrefix?: string;
	activeUrlIncludes?: string;
	keys?: string[];
}
