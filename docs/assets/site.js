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
      await navigator.clipboard.writeText(text);
      return true;
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
    const ok = await copyText(text);
    const original = button.getAttribute("data-label") || button.textContent.trim();
    button.setAttribute("data-label", original);
    button.textContent = ok ? "Copied" : "Copy failed";
    announce(ok ? "Copied to clipboard" : "Copy failed");
    window.setTimeout(() => {
      button.textContent = original;
    }, 1600);
  });

  /* --- Reveal on scroll ------------------------------------------------- */
  const revealables = [...doc.querySelectorAll(".reveal")];
  if (!reduceMotion.matches && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        }
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
    );
    revealables.forEach((item) => observer.observe(item));
  } else {
    revealables.forEach((item) => item.classList.add("is-visible"));
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

  /* --- Landing: replay the recorded audit fix --------------------------- */
  const audit = doc.querySelector("[data-audit]");
  if (audit) {
    const button = audit.querySelector("[data-audit-toggle]");
    const ring = audit.querySelector(".score-ring");
    const score = audit.querySelector("[data-audit-score]");
    const verdict = audit.querySelector("[data-audit-verdict]");
    const note = audit.querySelector("[data-audit-note]");
    const counts = audit.querySelector("[data-audit-counts]");
    let fixed = false;
    button.addEventListener("click", () => {
      fixed = !fixed;
      audit.classList.toggle("is-fixed", fixed);
      button.setAttribute("aria-pressed", String(fixed));
      if (fixed) {
        ring.style.setProperty("--score", "100");
        ring.style.setProperty("--ring-color", "var(--safe)");
        score.textContent = "100";
        verdict.textContent = "Clean audit";
        counts.textContent = "0 high · 0 medium · 0 low";
        note.textContent =
          "Recorded second run: both leaky columns dropped. The remaining features pass the audit (100/100).";
        note.classList.add("is-safe");
        button.textContent = "Replay the leaky version";
        announce("Clean audit: trust score 100 out of 100");
      } else {
        ring.style.setProperty("--score", "50");
        ring.style.setProperty("--ring-color", "var(--danger)");
        score.textContent = "50";
        verdict.textContent = "Leaky table";
        counts.textContent = "1 high · 1 medium · 0 low";
        note.textContent =
          "Prerecorded output: leaky_transactions.csv, target is_fraud, generated by leaklens.leakage.scan and score().";
        note.classList.remove("is-safe");
        button.textContent = "Apply the fix (recorded)";
        announce("Leaky audit: trust score 50 out of 100");
      }
    });
  }
})();
