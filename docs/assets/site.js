/* Shared site behaviour: navigation, copy buttons, reveals, docs scrollspy.
   No dependencies, no network calls. */
(() => {
  "use strict";

  const doc = document;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  /* --- Live region for copy / navigation feedback ----------------------- */
  const live = doc.createElement("p");
  live.className = "sr-only";
  live.setAttribute("role", "status");
  live.setAttribute("aria-live", "polite");
  live.style.cssText =
    "position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;border:0";
  doc.body.append(live);
  const announce = (message) => {
    live.textContent = "";
    window.setTimeout(() => {
      live.textContent = message;
    }, 20);
  };

  /* --- Sticky header ---------------------------------------------------- */
  const header = doc.querySelector("[data-header]");
  if (header) {
    const setScrolled = () => header.classList.toggle("is-scrolled", window.scrollY > 8);
    setScrolled();
    window.addEventListener("scroll", setScrolled, { passive: true });
  }

  /* --- Mobile navigation ------------------------------------------------ */
  const navToggle = doc.querySelector("[data-nav-toggle]");
  const nav = doc.getElementById("site-nav");
  if (navToggle && nav) {
    const setOpen = (open) => {
      navToggle.setAttribute("aria-expanded", String(open));
      nav.classList.toggle("is-open", open);
    };
    navToggle.addEventListener("click", () => {
      setOpen(navToggle.getAttribute("aria-expanded") !== "true");
    });
    nav.addEventListener("click", (event) => {
      if (event.target.closest("a")) setOpen(false);
    });
    doc.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && navToggle.getAttribute("aria-expanded") === "true") {
        setOpen(false);
        navToggle.focus();
      }
    });
    doc.addEventListener("click", (event) => {
      if (!nav.contains(event.target) && !navToggle.contains(event.target)) setOpen(false);
    });
  }

  /* --- Copy buttons ----------------------------------------------------- */
  const copyText = async (text) => {
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch {
        // Permission denied or an inactive document: use the selection fallback.
      }
    }
    const field = doc.createElement("textarea");
    field.value = text;
    field.setAttribute("readonly", "");
    field.style.position = "fixed";
    field.style.opacity = "0";
    doc.body.append(field);
    field.select();
    let ok = false;
    try {
      ok = doc.execCommand("copy");
    } catch {
      ok = false;
    }
    field.remove();
    return ok;
  };

  doc.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-copy]");
    if (!button) return;
    const selector = button.getAttribute("data-copy-target");
    const source = selector ? doc.querySelector(selector) : null;
    const text = source ? source.textContent.trim() : button.getAttribute("data-copy");
    if (!text) return;
    // Capture the label before awaiting: a second click inside the feedback
    // window would otherwise snapshot "Copied" and never restore the command.
    const original = button.dataset.original || button.textContent.trim();
    button.dataset.original = original;
    const ok = await copyText(text);
    if (button.copyTimer) window.clearTimeout(button.copyTimer);
    button.textContent = ok ? "Copied" : "Copy failed";
    announce(ok ? "Copied to clipboard" : "Copy failed");
    button.copyTimer = window.setTimeout(() => {
      button.textContent = original;
    }, 1600);
  });

  /* --- Reveal on scroll ------------------------------------------------- */
  const revealables = [...doc.querySelectorAll(".reveal")];
  const showAllReveals = () => {
    revealables.forEach((item) => item.classList.add("is-visible"));
  };
  if (!reduceMotion.matches && "IntersectionObserver" in window) {
    let sawIntersection = false;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            sawIntersection = true;
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        }
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
    );
    revealables.forEach((item) => observer.observe(item));
    // Failsafe only when no element ever intersected (paused document or a
    // renderer that never composites); otherwise keep reveal-on-scroll intact.
    window.setTimeout(() => {
      if (!sawIntersection) showAllReveals();
    }, 1500);
  } else {
    showAllReveals();
  }

  /* --- Docs: sidebar toggle, scrollspy and top link --------------------- */
  const docsToggle = doc.querySelector("[data-docs-toggle]");
  const docsNav = doc.querySelector("[data-docs-nav]");
  if (docsToggle && docsNav) {
    docsToggle.addEventListener("click", () => {
      const open = docsNav.classList.toggle("is-open");
      docsToggle.setAttribute("aria-expanded", String(open));
    });
    docsNav.addEventListener("click", (event) => {
      if (event.target.closest("a") && window.innerWidth <= 820) {
        docsNav.classList.remove("is-open");
        docsToggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  const docSections = [...doc.querySelectorAll("[data-doc-section]")];
  const docLinks = docsNav ? [...docsNav.querySelectorAll('a[href^="#"]')] : [];
  if (docSections.length && docLinks.length && "IntersectionObserver" in window) {
    const byId = new Map(docLinks.map((link) => [link.getAttribute("href").slice(1), link]));
    const spy = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        docLinks.forEach((link) => link.classList.remove("is-active"));
        const active = byId.get(visible.target.id);
        if (active) {
          active.classList.add("is-active");
          active.setAttribute("aria-current", "true");
        }
      },
      { rootMargin: "-20% 0px -60% 0px", threshold: [0.1, 0.4] }
    );
    docSections.forEach((section) => spy.observe(section));
  }

  /* --- Landing: preview a proposed audit change ------------------------- */
  const audit = doc.querySelector("[data-audit]");
  const auditButton = audit?.querySelector("[data-audit-toggle]");
  if (audit && auditButton) {
    const diff = audit.querySelector("[data-audit-diff]");
    auditButton.addEventListener("click", () => {
      const open = auditButton.getAttribute("aria-expanded") !== "true";
      auditButton.setAttribute("aria-expanded", String(open));
      if (diff) diff.hidden = !open;
      auditButton.textContent = open
        ? "Hide the proposed change"
        : "Preview the proposed change";
      announce(
        open
          ? "Proposed patch shown. Nothing was applied and no new scan ran."
          : "Proposed patch hidden."
      );
    });
  }
})();
