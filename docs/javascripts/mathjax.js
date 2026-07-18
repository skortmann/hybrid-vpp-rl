window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

/* Re-typeset after every page swap (Material instant navigation).
   Guard against the first emission, which fires before the MathJax
   library has loaded — an exception here would tear down the RxJS
   subscription and silently disable math on all later pages. */
document$.subscribe(() => {
  if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  }
});
