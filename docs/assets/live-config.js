/* Optional build-time configuration for docs/live.html.
 *
 * Leave apiBase empty to ship the static site in its honest "not connected"
 * state. A deployer can set it to the URL of a LeakLens public API, or users
 * can paste a URL into the page (stored in this browser only).
 *
 * Example for a locally running API:
 *   apiBase: "http://localhost:8000"
 *
 * Do not commit a URL for a backend you do not control: every scan result the
 * page displays would then come from that third party.
 */
window.LEAKLENS_LIVE_CONFIG = Object.freeze({
  apiBase: "",
  requestTimeoutMs: 60000,
});
