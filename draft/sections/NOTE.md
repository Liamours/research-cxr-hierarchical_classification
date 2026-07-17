consist of md files, with paths to figures or tables, root is: "C:\Users\lulay\Desktop\research-wbbs-pyradiomic\draft", so if figures are in "C:\Users\lulay\Desktop\research-wbbs-pyradiomic\draft\figures\figure-1.png", the path in the md file would be "figures/figure-1.png"

we place format-minimum md files (minimal "#*=-" characters), so that the content is the main focus

we are going to place these md into docs at the end later.

we usually divide per-sections, sometimes per-subsections when it is being done by different person.

naming: "{section}-{slug}.md" for a section-level file, "{section}.{subsection}-{slug}.md" for a subsection-level file, slug is always hyphen-separated (no underscores). section number alone (e.g. "2") is a short overview/intro for that section, kept separate from its subsection files (e.g. "2.1", "2.2") even if only one subsection exists so far. Example: "0-abstract.md", "1-introduction.md", "2-methodology.md", "2.1-dataset-and-annotation.md", "3.1-data-statistics.md", "4.1-error-patterns.md", "4.2-limitations.md".

writing rules:
1. do not add m dash
2. do not add ai common words
3. do not add overclaim
4. do not add a claim unless it is actually done or supported by a reference
5. do not denote an abbreviation unless its full words were stated before it, denote it as such: "The Lengthy Words First (TLWF)", then TLWF can be used afterward in the section
6. do not put information in brackets, prefer verbose phrasing instead, e.g. not "2,696 anterior and 2,332 posterior scans contain at least one annotated lesion (5,028 images used)" but "2,696 anterior and 2,332 posterior scans contain at least one annotated lesion, in total 5,028 images used"
7. do not make paragraphs a similar length, vary them
8. do not write short sentences without purpose, every sentence should carry weight
