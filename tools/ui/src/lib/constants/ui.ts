import { Folder, Search, Settings, SquarePen } from '@lucide/svelte';
import McpLogo from '$lib/components/app/mcp/McpLogo.svelte';
import type { Component } from 'svelte';
import { ROUTES } from './routes';

export const FORK_TREE_DEPTH_PADDING = 8;
export const SYSTEM_MESSAGE_PLACEHOLDER = 'System message';

export const ICON_STRIP_TRANSITION_DURATION = 150;
export const ICON_STRIP_TRANSITION_DELAY_MULTIPLIER = 50;

/** Max height for tool-result code blocks (json / source / diff / streaming code). */
export const MAX_HEIGHT_CODE_BLOCK = '22rem';

export interface DesktopIconStripItem {
	icon: Component;
	tooltip: string;
	route?: string;
	/** URL served outside the SPA - opened in a new tab, bypassing client-side routing. */
	externalHref?: string;
	activeRouteId?: string;
	activeRoutePrefix?: string;
	activeUrlIncludes?: string;
	keys?: string[];
}

export const SIDEBAR_ACTIONS_ITEMS: DesktopIconStripItem[] = [
	{ icon: SquarePen, tooltip: 'New chat', route: ROUTES.NEW_CHAT, keys: ['shift', 'cmd', 'o'] },
	{ icon: Search, tooltip: 'Search', keys: ['cmd', 'k'] },
	{
		icon: McpLogo,
		tooltip: 'MCP Servers',
		route: ROUTES.MCP_SERVERS,
		activeRouteId: '/mcp-servers'
	},
	{
		icon: Settings,
		tooltip: 'Settings',
		route: `${ROUTES.SETTINGS}/general`,
		activeUrlIncludes: '#/settings'
	}
];

/** Server-rendered file listing - only shown when the server reports `files_enabled`. */
export const SIDEBAR_FILES_ITEM: DesktopIconStripItem = {
	icon: Folder,
	tooltip: 'Files',
	externalHref: './files'
};
