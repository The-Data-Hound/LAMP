
def prokka(genome, outfolder):
    import os
    os.system('prokka %s --outdir %s --force'%(genome, outfolder))
    print('running prokka')
def card(genome, outfolder):
    print('card')

def mash():
    os.system('mash ')
    print('running mash')
def makemashdb():
    print('makemashdb')

def barrnap(genome, outfolder):
    from Bio import SeqIO
    import os
    os.system('barrnap -in %s -outseq %s'%(genome,outfolder+genome+'.barrnap.fna'))
    with open(outfolder+genome+'.barrnap.fna','r' )as h:
        for r in SeqIO.parse(h,'fasta'):
            if '16S' in str(r.description).upper():
                seq = str(r.seq)
    try:
        out = open(outfolder+genome+'.16s.fna','w')
        out.write('>%s\n%s\n'%(genome+'.16s.fna',seq))
        out.close()
        return outfolder+genome+'.16s.fna'
    except:
        print('No 16S Found')



def makeblastdb(file = 'bacteria.16SrRNA.fna',out = 'tlp16s'):
    import os
    h = open(file,'r').readlines()
    h=[i.replace(' ','_') for i in h]
    new = open(file,'w')
    [new.write(i) for i in h]
    new.close()
    os.system('makeblastdb -dbtype nucl -in %s -out %s'%(file,out))
    print('Making blastdb')
def blast16(seq_name, outfolder, db = 'tlp16s'):
    import pandas as pd
    import os
    os.system('blastn -db %s -query %s -out %s.tsv -outfmt 6'%(db, seq_name, seq_name))
    df = pd.read_csv(seq_name+'.tsv', header = None, sep = '\t')
    df.columns = ['qseqid','sseqid','pident','length','mismatch','gapopen','qstart','qend','sstart','send','evalue','bitscore']
    df.to_csv(seq_name+'.tsv',sep='\t',index = False)
    print('running blast 16s')
    return df
def get_blast_species(genome, outfolder, db = 'tlp16s'):
    import pandas as pd
    seq_name = barrnap(genome, outfolder)
    df = blast16(seq_name, outfolder, db)
    df = df.sort_values(by = 'pident', ascending = False).reset_index(drop=True)
    species=df.sseqid.iloc[0]
    dashes = [j for j, x in enumerate(species) if x == "_"]
    print(species[dashes[1]+1:dashes[2]]+' '+species[dashes[2]+1:dashes[3]])
    return species

def download_dbs(db ='tlp'):
    import os
    if db =='both':
        try:
            os.mkdir('lamp_dbs')
            os.chdir('lamp_dbs')
            os.system('wget ftp://ftp.ncbi.nlm.nih.gov:21/refseq/TargetedLoci/Bacteria/bacteria.16SrRNA.fna.gz')
            os.system('gunzip bacteria.16SrRNA.fna.gz')
            print('TLP Downloaded')
            print('Working on GTDB Warning: File is over 60 GB')
            os.system('wget https://data.ace.uq.edu.au/public/gtdb/data/releases/latest/genomic_files_reps/gtdb_genomes_reps.tar.gz')
            os.system('gunzip gtdb_genomes_reps.tar.gz')
            print('Finished GTDB')
            print('Dbs downloaded at %s'%os.getcwd())
        except:
            print('Something Went Wrong')
    if db =='gtdb':
        try:
            os.system('wget https://data.ace.uq.edu.au/public/gtdb/data/releases/latest/genomic_files_reps/gtdb_genomes_reps.tar.gz')
            os.system('gunzip gtdb_genomes_reps.tar.gz')
            print('Finished GTDB')
            print('Dbs downloaded at %s'%os.getcwd())
        except:
            print('Something Went Wrong')
    if db =='tlp':
        try:
            os.mkdir('lamp_dbs')
            os.chdir('lamp_dbs')
            os.system('wget ftp://ftp.ncbi.nlm.nih.gov:21/refseq/TargetedLoci/Bacteria/bacteria.16SrRNA.fna.gz')
            os.system('gunzip bacteria.16SrRNA.fna.gz')
            print('TLP Downloaded')
            print('Dbs downloaded at %s'%os.getcwd())
        except:
            print('Something Went Wrong')
